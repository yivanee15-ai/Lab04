//############################################################################
//   ICLAB 2026 Fall  Lab04 : Simplified Flash Attention (F_ATTN)
//   File Name   : F_ATTN.v
//   Module Name : F_ATTN
//
//   Architecture (area-reduced version: 3 exp units, 2-stage output pipeline):
//     * K,Q,V (64 words each) and out_weight (16 words) are stored while in_valid is high.
//     * Loop order: head (2) -> Q block (8) -> K/V block (8); one 2x2 tile per 7 cycles
//         T1 : prod = q*k (8 mult), s_raw = prod0 + prod1 (4 add)
//         T2 : s = s_raw / sqrt2                  (4 div)
//         T3 : rowmax, m_new = max(m, rowmax)     (4 cmp)
//              dm = m - m_new, ds = s - m_new     (6 add with negated operand)
//         E1 : alpha0 = exp(dm0), alpha1 = exp(dm1), p00 = exp(ds00)   (3 exp, alpha = 0 on the first tile)
//         E2 : p01, p10, p11 = exp(ds01), exp(ds10), exp(ds11)         (the same 3 exp units)
//         T5 : pv = p0*v0 + p1*v1 (8 mult, 4 add), psum = p0 + p1 (2 add)
//         T6 : acc = alpha*acc + pv, l = alpha*l + psum (6 mult, 6 add)
//       after the 8th K/V block:  FIN : head_out = acc / l (4 div)
//     * OUT : every cycle one result  ((h0*w0 + h1*w1) + h2*w2) + h3*w3 , 64 cycles,
//             2-stage pipeline: O1 = 4 mult + (p0+p1), O2 = (+p2) + p3
//   The same floating point units are shared between the states (operand muxes below).
//   Operation order matches 00_TESTBED/gen_pattern.py (online_f32).
//############################################################################

module F_ATTN(
    clk,
    rst_n,
    in_valid,
    Q,
    K,
    V,
    out_weight,
    out_valid,
    out
);

input         clk, rst_n, in_valid;
input  [31:0] Q, K, V, out_weight;
output reg        out_valid;
output reg [31:0] out;

//---------------------------------------------------------------------
//  parameters
//---------------------------------------------------------------------
parameter SIG_W  = 23;
parameter EXP_W  = 8;
parameter IEEE_C = 0;

localparam [31:0] SQRT2 = 32'h3FB504F3;     // sqrt(2), constant given in the lab PDF
localparam [31:0] FP0   = 32'h00000000;

localparam [3:0] S_LOAD = 4'd0,
                 S_T1   = 4'd1,
                 S_T2   = 4'd2,
                 S_T3   = 4'd3,
                 S_E1   = 4'd4,
                 S_E2   = 4'd5,
                 S_T5   = 4'd6,
                 S_T6   = 4'd7,
                 S_FIN  = 4'd8,
                 S_OUT  = 4'd9,
                 S_FLUSH= 4'd10,
                 S_DONE = 4'd11;

//---------------------------------------------------------------------
//  registers
//---------------------------------------------------------------------
reg [3:0]  st;
reg [5:0]  cnt;
reg [2:0]  bj, bi;          // K/V block index, Q block index
reg        hd;              // head index

reg [31:0] q_r  [0:63];
reg [31:0] k_r  [0:63];
reg [31:0] v_r  [0:63];
reg [31:0] w_r  [0:15];
reg [31:0] ho_r [0:63];     // head outputs, 16 tokens x 4 dims

reg [31:0] sr00, sr01, sr10, sr11;          // raw scores
reg [31:0] s00,  s01,  s10,  s11;           // scaled scores
reg [31:0] ds00, ds01, ds10, ds11;          // s - m_new
reg [31:0] dm0,  dm1;                       // m_old - m_new
reg [31:0] p00,  p01,  p10,  p11;           // exp(s - m_new)
reg [31:0] al0,  al1;                       // alpha
reg [31:0] pv00, pv01, pv10, pv11;          // P x V tile
reg [31:0] ps0,  ps1;                       // row sums of P
reg [31:0] ac00, ac01, ac10, ac11;          // O accumulator
reg [31:0] ll0,  ll1;                       // softmax denominator
reg [31:0] mx0,  mx1;                       // running max
reg [31:0] ra01, rp2, rp3;                  // output pipeline registers

wire first = (bj == 3'd0);

//---------------------------------------------------------------------
//  operand reads
//---------------------------------------------------------------------
wire [31:0] qr0k0 = q_r[{bi, 1'b0, hd, 1'b0}];
wire [31:0] qr0k1 = q_r[{bi, 1'b0, hd, 1'b1}];
wire [31:0] qr1k0 = q_r[{bi, 1'b1, hd, 1'b0}];
wire [31:0] qr1k1 = q_r[{bi, 1'b1, hd, 1'b1}];

wire [31:0] kc0k0 = k_r[{bj, 1'b0, hd, 1'b0}];
wire [31:0] kc0k1 = k_r[{bj, 1'b0, hd, 1'b1}];
wire [31:0] kc1k0 = k_r[{bj, 1'b1, hd, 1'b0}];
wire [31:0] kc1k1 = k_r[{bj, 1'b1, hd, 1'b1}];

wire [31:0] vc0d0 = v_r[{bj, 1'b0, hd, 1'b0}];
wire [31:0] vc0d1 = v_r[{bj, 1'b0, hd, 1'b1}];
wire [31:0] vc1d0 = v_r[{bj, 1'b1, hd, 1'b0}];
wire [31:0] vc1d1 = v_r[{bj, 1'b1, hd, 1'b1}];

// output projection operands: token = cnt[5:2], out column = cnt[1:0]
wire [31:0] hk0 = ho_r[{cnt[5:2], 2'd0}];
wire [31:0] hk1 = ho_r[{cnt[5:2], 2'd1}];
wire [31:0] hk2 = ho_r[{cnt[5:2], 2'd2}];
wire [31:0] hk3 = ho_r[{cnt[5:2], 2'd3}];
wire [31:0] wk0 = w_r[{cnt[1:0], 2'd0}];
wire [31:0] wk1 = w_r[{cnt[1:0], 2'd1}];
wire [31:0] wk2 = w_r[{cnt[1:0], 2'd2}];
wire [31:0] wk3 = w_r[{cnt[1:0], 2'd3}];

function [31:0] neg;
    input [31:0] x;
    begin
        neg = {~x[31], x[30:0]};
    end
endfunction

//---------------------------------------------------------------------
//  floating point units (shared between states)
//---------------------------------------------------------------------
reg  [31:0] mua [0:7];
reg  [31:0] mub [0:7];
wire [31:0] muz [0:7];

reg  [31:0] ada [0:5];
reg  [31:0] adb [0:5];
wire [31:0] adz [0:5];

reg  [31:0] dva [0:3];
reg  [31:0] dvb [0:3];
wire [31:0] dvz [0:3];

reg  [31:0] exa [0:2];
wire [31:0] exz [0:2];

wire [31:0] cma [0:3];
wire [31:0] cmb [0:3];
wire        cmg [0:3];

genvar g;
generate
    for (g = 0; g < 8; g = g + 1) begin : G_MUL
        DW_fp_mult #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C))
            u_mul (.a(mua[g]), .b(mub[g]), .rnd(3'b000), .z(muz[g]), .status());
    end
    for (g = 0; g < 6; g = g + 1) begin : G_ADD
        DW_fp_add #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C))
            u_add (.a(ada[g]), .b(adb[g]), .rnd(3'b000), .z(adz[g]), .status());
    end
    for (g = 0; g < 4; g = g + 1) begin : G_DIV
        DW_fp_div #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C))
            u_div (.a(dva[g]), .b(dvb[g]), .rnd(3'b000), .z(dvz[g]), .status());
    end
    for (g = 0; g < 3; g = g + 1) begin : G_EXP
        DW_fp_exp #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C))
            u_exp (.a(exa[g]), .z(exz[g]), .status());
    end
    for (g = 0; g < 4; g = g + 1) begin : G_CMP
        DW_fp_cmp #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C))
            u_cmp (.a(cma[g]), .b(cmb[g]), .zctr(1'b0), .aeqb(), .altb(), .agtb(cmg[g]),
                   .unordered(), .z0(), .z1(), .status0(), .status1());
    end
endgenerate

//---------------------------------------------------------------------
//  max tracking (T3): rowmax then running max, chained compares
//---------------------------------------------------------------------
assign cma[0] = s01;    assign cmb[0] = s00;       // s01 > s00 ?
assign cma[1] = s11;    assign cmb[1] = s10;
wire [31:0] rmax0 = cmg[0] ? s01 : s00;
wire [31:0] rmax1 = cmg[1] ? s11 : s10;
assign cma[2] = rmax0;  assign cmb[2] = mx0;       // rmax > m_old ?
assign cma[3] = rmax1;  assign cmb[3] = mx1;
wire [31:0] mnew0 = first ? rmax0 : (cmg[2] ? rmax0 : mx0);
wire [31:0] mnew1 = first ? rmax1 : (cmg[3] ? rmax1 : mx1);

//---------------------------------------------------------------------
//  operand muxes
//---------------------------------------------------------------------
integer n;
always @(*) begin
    for (n = 0; n < 8; n = n + 1) begin
        mua[n] = FP0;
        mub[n] = FP0;
    end
    for (n = 0; n < 6; n = n + 1) begin
        ada[n] = FP0;
        adb[n] = FP0;
    end
    for (n = 0; n < 4; n = n + 1) begin
        dva[n] = FP0;
        dvb[n] = FP0;
    end
    for (n = 0; n < 3; n = n + 1) exa[n] = FP0;

    case (st)
    S_T1: begin
        // prod[r][c][k] = q[r][k] * k[c][k]
        mua[0] = qr0k0; mub[0] = kc0k0;     mua[1] = qr0k1; mub[1] = kc0k1;
        mua[2] = qr0k0; mub[2] = kc1k0;     mua[3] = qr0k1; mub[3] = kc1k1;
        mua[4] = qr1k0; mub[4] = kc0k0;     mua[5] = qr1k1; mub[5] = kc0k1;
        mua[6] = qr1k0; mub[6] = kc1k0;     mua[7] = qr1k1; mub[7] = kc1k1;
        ada[0] = muz[0]; adb[0] = muz[1];
        ada[1] = muz[2]; adb[1] = muz[3];
        ada[2] = muz[4]; adb[2] = muz[5];
        ada[3] = muz[6]; adb[3] = muz[7];
    end
    S_T2: begin
        dva[0] = sr00; dvb[0] = SQRT2;
        dva[1] = sr01; dvb[1] = SQRT2;
        dva[2] = sr10; dvb[2] = SQRT2;
        dva[3] = sr11; dvb[3] = SQRT2;
    end
    S_T3: begin
        ada[0] = s00; adb[0] = neg(mnew0);
        ada[1] = s01; adb[1] = neg(mnew0);
        ada[2] = s10; adb[2] = neg(mnew1);
        ada[3] = s11; adb[3] = neg(mnew1);
        ada[4] = mx0; adb[4] = neg(mnew0);
        ada[5] = mx1; adb[5] = neg(mnew1);
    end
    S_E1: begin
        exa[0] = dm0;  exa[1] = dm1;  exa[2] = ds00;
    end
    S_E2: begin
        exa[0] = ds01; exa[1] = ds10; exa[2] = ds11;
    end
    S_T5: begin
        // pv[r][d] = p[r][0]*v[0][d] + p[r][1]*v[1][d]
        mua[0] = p00; mub[0] = vc0d0;       mua[1] = p01; mub[1] = vc1d0;
        mua[2] = p00; mub[2] = vc0d1;       mua[3] = p01; mub[3] = vc1d1;
        mua[4] = p10; mub[4] = vc0d0;       mua[5] = p11; mub[5] = vc1d0;
        mua[6] = p10; mub[6] = vc0d1;       mua[7] = p11; mub[7] = vc1d1;
        ada[0] = muz[0]; adb[0] = muz[1];
        ada[1] = muz[2]; adb[1] = muz[3];
        ada[2] = muz[4]; adb[2] = muz[5];
        ada[3] = muz[6]; adb[3] = muz[7];
        ada[4] = p00;    adb[4] = p01;
        ada[5] = p10;    adb[5] = p11;
    end
    S_T6: begin
        mua[0] = al0; mub[0] = ac00;
        mua[1] = al0; mub[1] = ac01;
        mua[2] = al1; mub[2] = ac10;
        mua[3] = al1; mub[3] = ac11;
        mua[4] = al0; mub[4] = ll0;
        mua[5] = al1; mub[5] = ll1;
        ada[0] = muz[0]; adb[0] = pv00;
        ada[1] = muz[1]; adb[1] = pv01;
        ada[2] = muz[2]; adb[2] = pv10;
        ada[3] = muz[3]; adb[3] = pv11;
        ada[4] = muz[4]; adb[4] = ps0;
        ada[5] = muz[5]; adb[5] = ps1;
    end
    S_FIN: begin
        dva[0] = ac00; dvb[0] = ll0;
        dva[1] = ac01; dvb[1] = ll0;
        dva[2] = ac10; dvb[2] = ll1;
        dva[3] = ac11; dvb[3] = ll1;
    end
    S_OUT: begin
        mua[0] = hk0; mub[0] = wk0;
        mua[1] = hk1; mub[1] = wk1;
        mua[2] = hk2; mub[2] = wk2;
        mua[3] = hk3; mub[3] = wk3;
        ada[0] = muz[0]; adb[0] = muz[1];
        ada[1] = ra01;   adb[1] = rp2;
        ada[2] = adz[1]; adb[2] = rp3;
    end
    S_FLUSH: begin
        ada[1] = ra01;   adb[1] = rp2;
        ada[2] = adz[1]; adb[2] = rp3;
    end
    default: ;
    endcase
end

//---------------------------------------------------------------------
//  control (asynchronous active-low reset)
//---------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        st        <= S_LOAD;
        cnt       <= 6'd0;
        bj        <= 3'd0;
        bi        <= 3'd0;
        hd        <= 1'b0;
        out_valid <= 1'b0;
        out       <= 32'd0;
        ac00 <= FP0; ac01 <= FP0; ac10 <= FP0; ac11 <= FP0;
        ll0  <= FP0; ll1  <= FP0;
    end
    else begin
        case (st)
        S_LOAD: begin
            if (in_valid) begin
                cnt <= cnt + 6'd1;
                if (cnt == 6'd63) begin
                    cnt <= 6'd0;
                    bj  <= 3'd0;
                    bi  <= 3'd0;
                    hd  <= 1'b0;
                    st  <= S_T1;
                end
            end
        end
        S_T1: st <= S_T2;
        S_T2: st <= S_T3;
        S_T3: st <= S_E1;
        S_E1: st <= S_E2;
        S_E2: st <= S_T5;
        S_T5: st <= S_T6;
        S_T6: begin
            ac00 <= adz[0]; ac01 <= adz[1]; ac10 <= adz[2]; ac11 <= adz[3];
            ll0  <= adz[4]; ll1  <= adz[5];
            if (bj == 3'd7) st <= S_FIN;
            else begin
                bj <= bj + 3'd1;
                st <= S_T1;
            end
        end
        S_FIN: begin
            ac00 <= FP0; ac01 <= FP0; ac10 <= FP0; ac11 <= FP0;
            ll0  <= FP0; ll1  <= FP0;
            bj   <= 3'd0;
            if ({hd, bi} == 4'b1111) begin
                cnt <= 6'd0;
                st  <= S_OUT;
            end
            else begin
                {hd, bi} <= {hd, bi} + 4'd1;
                st <= S_T1;
            end
        end
        S_OUT: begin                    // stage O1 for index cnt, stage O2 for index cnt-1
            out_valid <= (cnt != 6'd0);
            out       <= (cnt != 6'd0) ? adz[2] : 32'd0;
            cnt       <= cnt + 6'd1;
            if (cnt == 6'd63) st <= S_FLUSH;
        end
        S_FLUSH: begin                  // stage O2 for the last index
            out_valid <= 1'b1;
            out       <= adz[2];
            st        <= S_DONE;
        end
        S_DONE: begin
            out_valid <= 1'b0;
            out       <= 32'd0;
            cnt       <= 6'd0;
            st        <= S_LOAD;
        end
        default: st <= S_LOAD;
        endcase
    end
end

//---------------------------------------------------------------------
//  datapath registers (no reset needed)
//---------------------------------------------------------------------
always @(posedge clk) begin
    if (st == S_LOAD && in_valid) begin
        q_r[cnt] <= Q;
        k_r[cnt] <= K;
        v_r[cnt] <= V;
        if (cnt < 6'd16) w_r[cnt[3:0]] <= out_weight;
    end

    case (st)
    S_T1: begin
        sr00 <= adz[0]; sr01 <= adz[1]; sr10 <= adz[2]; sr11 <= adz[3];
    end
    S_T2: begin
        s00 <= dvz[0]; s01 <= dvz[1]; s10 <= dvz[2]; s11 <= dvz[3];
    end
    S_T3: begin
        ds00 <= adz[0]; ds01 <= adz[1]; ds10 <= adz[2]; ds11 <= adz[3];
        dm0  <= first ? FP0 : adz[4];
        dm1  <= first ? FP0 : adz[5];
        mx0  <= mnew0;
        mx1  <= mnew1;
    end
    S_E1: begin
        al0 <= first ? FP0 : exz[0];
        al1 <= first ? FP0 : exz[1];
        p00 <= exz[2];
    end
    S_E2: begin
        p01 <= exz[0]; p10 <= exz[1]; p11 <= exz[2];
    end
    S_T5: begin
        pv00 <= adz[0]; pv01 <= adz[1]; pv10 <= adz[2]; pv11 <= adz[3];
        ps0  <= adz[4]; ps1  <= adz[5];
    end
    S_OUT: begin
        ra01 <= adz[0];
        rp2  <= muz[2];
        rp3  <= muz[3];
    end
    S_FIN: begin
        ho_r[{bi, 1'b0, hd, 1'b0}] <= dvz[0];
        ho_r[{bi, 1'b0, hd, 1'b1}] <= dvz[1];
        ho_r[{bi, 1'b1, hd, 1'b0}] <= dvz[2];
        ho_r[{bi, 1'b1, hd, 1'b1}] <= dvz[3];
    end
    default: ;
    endcase
end

endmodule
