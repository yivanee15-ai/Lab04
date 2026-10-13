//############################################################################
//   ICLAB 2026 Fall  Lab04 : Simplified Flash Attention (F_ATTN)
//   File Name   : F_ATTN.v
//   Module Name : F_ATTN
//
//   Architecture : compute while receiving, one pipelined row engine shared by both heads
//
//   * K,Q,V (64 words each) and out_weight (16 words) are stored while in_valid is high.
//   * Block b (tokens 2b,2b+1) is complete after 8(b+1) input beats.  As soon as it is
//     complete, every tile that needs it is issued:
//         phase b :  tiles (i, j=b) for i = 0..b-1   (one more K/V block for the older chains)
//                    tiles (i=b, j) for j = 0..b     (new Q block against all arrived K/V blocks)
//     so for every (head, Q block) chain the K/V blocks are still visited in the order
//     j = 0..7 (same rounding as the reference model).
//   * A tile is issued as two row-tasks (row 0, row 1) per head, heads alternate:
//         (hd0,r0) (hd0,r1) (hd1,r0) (hd1,r1)   -> 1 row-task per cycle, same chain every >=4 cycles
//   * Row-task pipeline (every stage has its own units, no operand muxes):
//         F1 : prod = q*k (4 mult), s_raw = prod0 + prod1 (2 add); QK_DP2=1 uses 2 DW_fp_dp2 instead
//         F2 : s = s_raw / sqrt2                      (2 div)
//         R1 : rmax, m_new = max(m, rmax)             (2 cmp)   m state read / written here
//              ds = s - m_new, dm = m - m_new         (3 add, negated operand)
//         R2 : alpha = exp(dm) (0 on the first tile), p = exp(ds)   (3 exp)
//         R3 : alpha*acc, alpha*l (3 mult), pv = p0*v0 + p1*v1 (4 mult, 2 add), psum = p0 + p1 (1 add)
//         R4 : acc = alpha*acc + pv, l = alpha*l + psum (3 add)     acc / l state written here
//     State per (head, Q block, row): m, l, acc[2]  (32 + 32 + 64 words).
//   * Output engine (dedicated units, overlaps the last tiles):
//         for every token: head_out = acc / l (4 divides, one per cycle, prepared one token ahead)
//         O1 : 4 mult + (p0 + p1)       O2 : (+p2) + p3      -> out, 1 word per cycle, 64 cycles
//     It starts a fixed number of cycles after phase 7 starts issuing (P7_DLY); the schedule is
//     data independent, so this is exact (P7_DLY-1 produces wrong results).
//   Operation order matches 00_TESTBED/PATTERN.v (golden) and gen_pattern.py (online_f32).
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
//  parameters / helper macros
//---------------------------------------------------------------------
parameter SIG_W    = 23;
parameter EXP_W    = 8;
parameter IEEE_C   = 0;
parameter EXP_ARCH = 0;
parameter QK_DP2   = 1;     // 0: q0*k0 + q1*k1 with 2 mult + 1 add, 1: one DW_fp_dp2 (single rounding)
parameter DP_ARCH  = 0;     // arch_type of DW_fp_dp2
parameter W_T      = 0;     // 1: out_weight is delivered already transposed (Final = Concat x W)
parameter Q_PRE    = 1;     // 1: Q is divided by sqrt2 as it arrives (matches TA golden), 0: S divided by sqrt2 after QK
parameter P7_DLY   = 4;     // output start = first phase-7 issue + 2 + P7_DLY cycles

localparam [31:0] SQRT2 = 32'h3FB504F3;     // sqrt(2), constant given in the lab PDF
localparam [31:0] FP0   = 32'h00000000;

`define FA_MUL(U_, A_, B_, Z_) \
    DW_fp_mult #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C)) \
        U_ (.a(A_), .b(B_), .rnd(3'b000), .z(Z_), .status());
`define FA_ADD(U_, A_, B_, Z_) \
    DW_fp_add #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C)) \
        U_ (.a(A_), .b(B_), .rnd(3'b000), .z(Z_), .status());
`define FA_DIV(U_, A_, B_, Z_) \
    DW_fp_div #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C)) \
        U_ (.a(A_), .b(B_), .rnd(3'b000), .z(Z_), .status());
`define FA_DP2(U_, A_, B_, C_, D_, Z_) \
    DW_fp_dp2 #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C), .arch_type(DP_ARCH)) \
        U_ (.a(A_), .b(B_), .c(C_), .d(D_), .rnd(3'b000), .z(Z_), .status());
`define FA_EXP(U_, A_, Z_) \
    DW_fp_exp #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C), .arch(EXP_ARCH)) \
        U_ (.a(A_), .z(Z_), .status());
`define FA_GT(U_, A_, B_, G_) \
    DW_fp_cmp #(.sig_width(SIG_W), .exp_width(EXP_W), .ieee_compliance(IEEE_C)) \
        U_ (.a(A_), .b(B_), .zctr(1'b0), .aeqb(), .altb(), .agtb(G_), .unordered(), \
            .z0(), .z1(), .status0(), .status1());

function [31:0] neg;
    input [31:0] x;
    begin
        neg = {~x[31], x[30:0]};
    end
endfunction

//---------------------------------------------------------------------
//  input storage
//---------------------------------------------------------------------
reg [31:0] q_r [0:63];
reg [31:0] k_r [0:63];
reg [31:0] v_r [0:63];
reg [31:0] w_r [0:15];

reg [6:0]  ld_cnt;                      // input beats received so far (0..64)
reg [6:0]  ld_q;                        // ld_cnt delayed one cycle (Q is stored one cycle later)

reg [31:0] qin_r;                       // Q delayed one beat, then (optionally) scaled
reg [5:0]  qin_i;
reg        qin_v;
wire [31:0] qin_s;
`FA_DIV(u_q_d, qin_r, SQRT2, qin_s)

always @(posedge clk) begin
    qin_r <= Q;
    qin_i <= ld_cnt[5:0];
    qin_v <= in_valid;
    if (qin_v) q_r[qin_i] <= (Q_PRE != 0) ? qin_s : qin_r;
    if (in_valid) begin
        k_r[ld_cnt[5:0]] <= K;
        v_r[ld_cnt[5:0]] <= V;
        if (ld_cnt < 7'd16) w_r[ld_cnt[3:0]] <= out_weight;
    end
end

//---------------------------------------------------------------------
//  state of every (head, Q block, row) chain : index = {hd, i, r}
//---------------------------------------------------------------------
reg [31:0] m_arr   [0:31];
reg [31:0] l_arr   [0:31];
reg [31:0] acc_arr [0:63];              // index {hd, i, r, d}

//---------------------------------------------------------------------
//  tile sequencer
//---------------------------------------------------------------------
reg [2:0] sb;                           // phase = newest arrived block
reg [3:0] sidx;                         // position inside the phase (0..2b)
reg       shd;                          // head
reg       sq_r;                         // row
reg       sdone;

wire       col   = (sidx < {1'b0, sb});             // column tile: (i = idx, j = b)
wire [2:0] ti    = col ? sidx[2:0] : sb;
wire [3:0] rdiff = sidx - {1'b0, sb};
wire [2:0] tj    = col ? sb : rdiff[2:0];
wire       issue = !sdone && (ld_q >= {(sb == 3'd7), (sb + 3'd1) & 3'b111, 3'b000});

//---------------------------------------------------------------------
//  F1 : q*k, add
//---------------------------------------------------------------------
wire [31:0] q0  = q_r[{ti, sq_r, shd, 1'b0}];
wire [31:0] q1  = q_r[{ti, sq_r, shd, 1'b1}];
wire [31:0] k00 = k_r[{tj, 1'b0, shd, 1'b0}];
wire [31:0] k01 = k_r[{tj, 1'b0, shd, 1'b1}];
wire [31:0] k10 = k_r[{tj, 1'b1, shd, 1'b0}];
wire [31:0] k11 = k_r[{tj, 1'b1, shd, 1'b1}];

wire [31:0] f1_s0, f1_s1;
generate
    if (QK_DP2 != 0) begin : G_QK_FUSED
        `FA_DP2(u_f1_d0, q0, k00, q1, k01, f1_s0)
        `FA_DP2(u_f1_d1, q0, k10, q1, k11, f1_s1)
    end
    else begin : G_QK_SEP
        wire [31:0] f1_p00, f1_p01, f1_p10, f1_p11;
        `FA_MUL(u_f1_m00, q0, k00, f1_p00)
        `FA_MUL(u_f1_m01, q1, k01, f1_p01)
        `FA_MUL(u_f1_m10, q0, k10, f1_p10)
        `FA_MUL(u_f1_m11, q1, k11, f1_p11)
        `FA_ADD(u_f1_a0, f1_p00, f1_p01, f1_s0)
        `FA_ADD(u_f1_a1, f1_p10, f1_p11, f1_s1)
    end
endgenerate

// pipeline control: valid bit and tag {hd, i, r, j} per stage
reg        va, vb, vc, vd, ve;
reg [7:0]  ta, tb, tc, td, te;

reg [31:0] sc0, sc1;                    // after F1
reg [31:0] s0,  s1;                     // after F2
reg [31:0] ds0, ds1, dm;                // after R1
reg [31:0] pe0, pe1, al;                // after R2
reg [31:0] am0, am1, lm, pv0, pv1, ps;  // after R3

//---------------------------------------------------------------------
//  F2 : divide by sqrt2
//---------------------------------------------------------------------
wire [31:0] f2_s0, f2_s1;
generate if (Q_PRE != 0) begin : G_NOSC
    assign f2_s0 = sc0;
    assign f2_s1 = sc1;
end else begin : G_SC
    `FA_DIV(u_f2_d0, sc0, SQRT2, f2_s0)
    `FA_DIV(u_f2_d1, sc1, SQRT2, f2_s1)
end endgenerate

//---------------------------------------------------------------------
//  R1 : running max, subtract
//---------------------------------------------------------------------
wire        first_b = (tb[2:0] == 3'd0);
wire [31:0] m_rd    = m_arr[tb[7:3]];
wire        g_a, g_b;
`FA_GT(u_r1_ca, s1, s0, g_a)
wire [31:0] rmax = g_a ? s1 : s0;
`FA_GT(u_r1_cb, rmax, m_rd, g_b)
wire [31:0] mnew = (first_b | g_b) ? rmax : m_rd;
wire [31:0] nmn  = neg(mnew);
wire [31:0] r1_ds0, r1_ds1, r1_dm;
`FA_ADD(u_r1_a0, s0,   nmn, r1_ds0)
`FA_ADD(u_r1_a1, s1,   nmn, r1_ds1)
`FA_ADD(u_r1_a2, m_rd, nmn, r1_dm)

//---------------------------------------------------------------------
//  R2 : exp
//---------------------------------------------------------------------
wire        first_c = (tc[2:0] == 3'd0);
wire [31:0] r2_al, r2_p0, r2_p1;
`FA_EXP(u_r2_e0, dm,  r2_al)
`FA_EXP(u_r2_e1, ds0, r2_p0)
`FA_EXP(u_r2_e2, ds1, r2_p1)

//---------------------------------------------------------------------
//  R3 : alpha*state, p*v
//---------------------------------------------------------------------
wire        first_d = (td[2:0] == 3'd0);
wire [31:0] ag0 = first_d ? FP0 : acc_arr[{td[7:3], 1'b0}];
wire [31:0] ag1 = first_d ? FP0 : acc_arr[{td[7:3], 1'b1}];
wire [31:0] lg  = first_d ? FP0 : l_arr[td[7:3]];

wire [31:0] v00 = v_r[{td[2:0], 1'b0, td[7], 1'b0}];    // v[c=0][d=0]
wire [31:0] v01 = v_r[{td[2:0], 1'b0, td[7], 1'b1}];    // v[c=0][d=1]
wire [31:0] v10 = v_r[{td[2:0], 1'b1, td[7], 1'b0}];    // v[c=1][d=0]
wire [31:0] v11 = v_r[{td[2:0], 1'b1, td[7], 1'b1}];    // v[c=1][d=1]

wire [31:0] r3_am0, r3_am1, r3_lm, r3_pa, r3_pb, r3_pc, r3_pd, r3_pv0, r3_pv1, r3_ps;
`FA_MUL(u_r3_m0, al,  ag0, r3_am0)
`FA_MUL(u_r3_m1, al,  ag1, r3_am1)
`FA_MUL(u_r3_m2, al,  lg,  r3_lm)
`FA_MUL(u_r3_m3, pe0, v00, r3_pa)
`FA_MUL(u_r3_m4, pe1, v10, r3_pb)
`FA_MUL(u_r3_m5, pe0, v01, r3_pc)
`FA_MUL(u_r3_m6, pe1, v11, r3_pd)
`FA_ADD(u_r3_a0, r3_pa, r3_pb, r3_pv0)
`FA_ADD(u_r3_a1, r3_pc, r3_pd, r3_pv1)
`FA_ADD(u_r3_a2, pe0,   pe1,   r3_ps)

//---------------------------------------------------------------------
//  R4 : state update
//---------------------------------------------------------------------
wire [31:0] r4_acc0, r4_acc1, r4_l;
`FA_ADD(u_r4_a0, am0, pv0, r4_acc0)
`FA_ADD(u_r4_a1, am1, pv1, r4_acc1)
`FA_ADD(u_r4_a2, lm,  ps,  r4_l)

always @(posedge clk) begin
    // data registers: no enable, garbage while the stage is empty
    sc0 <= f1_s0;  sc1 <= f1_s1;
    s0  <= f2_s0;  s1  <= f2_s1;
    ds0 <= r1_ds0; ds1 <= r1_ds1;  dm <= first_b ? FP0 : r1_dm;
    pe0 <= r2_p0;  pe1 <= r2_p1;   al <= first_c ? FP0 : r2_al;
    am0 <= r3_am0; am1 <= r3_am1;  lm <= r3_lm;
    pv0 <= r3_pv0; pv1 <= r3_pv1;  ps <= r3_ps;

    ta <= {shd, ti, sq_r, tj};
    tb <= ta;
    tc <= tb;
    td <= tc;
    te <= td;

    // state arrays
    if (vb) m_arr[tb[7:3]] <= mnew;
    if (ve) begin
        acc_arr[{te[7:3], 1'b0}] <= r4_acc0;
        acc_arr[{te[7:3], 1'b1}] <= r4_acc1;
        l_arr[te[7:3]]           <= r4_l;
    end
end

//---------------------------------------------------------------------
//  output engine
//---------------------------------------------------------------------
localparam [2:0] O_IDLE  = 3'd0,
                 O_PREP  = 3'd1,
                 O_RUN   = 3'd2,
                 O_FLUSH = 3'd3,
                 O_DONE  = 3'd4;

reg [2:0]  ost;
reg [1:0]  pc;                          // prepare counter (start-up only)
reg [5:0]  oc;                          // output word counter
reg        p7_on;
reg [3:0]  p7c;
reg        ov2;

reg [31:0] ho0, ho1, ho2, ho3;          // head outputs of the token being projected
reg [31:0] hn0, hn1, hn2;               // next token, being prepared
reg [31:0] ra01, rp2, rp3;

wire [1:0] pcur = (ost == O_PREP) ? pc : oc[1:0];
wire [3:0] tp   = (ost == O_PREP) ? 4'd0 : (oc[5:2] + 4'd1);
wire [4:0] eo   = {pcur[1], tp};        // {hd, token}
wire [31:0] o_acc = acc_arr[{eo, pcur[0]}];
wire [31:0] o_l   = l_arr[eo];
wire [31:0] o_dz;
`FA_DIV(u_o_d, o_acc, o_l, o_dz)

// W_T = 0 : Final[t][c] = sum_k Concat[t][k] * W[c][k]   (W row c at index c*4+k)
// W_T = 1 : Final[t][c] = sum_k Concat[t][k] * W[k][c]   (index k*4+c)
wire [31:0] wk0 = w_r[(W_T != 0) ? {2'd0, oc[1:0]} : {oc[1:0], 2'd0}];
wire [31:0] wk1 = w_r[(W_T != 0) ? {2'd1, oc[1:0]} : {oc[1:0], 2'd1}];
wire [31:0] wk2 = w_r[(W_T != 0) ? {2'd2, oc[1:0]} : {oc[1:0], 2'd2}];
wire [31:0] wk3 = w_r[(W_T != 0) ? {2'd3, oc[1:0]} : {oc[1:0], 2'd3}];
wire [31:0] o_m0, o_m1, o_m2, o_m3, o_a01, o_b, o_c;
`FA_MUL(u_o_m0, ho0, wk0, o_m0)
`FA_MUL(u_o_m1, ho1, wk1, o_m1)
`FA_MUL(u_o_m2, ho2, wk2, o_m2)
`FA_MUL(u_o_m3, ho3, wk3, o_m3)
`FA_ADD(u_o_a0, o_m0, o_m1, o_a01)
`FA_ADD(u_o_a1, ra01, rp2,  o_b)
`FA_ADD(u_o_a2, o_b,  rp3,  o_c)

always @(posedge clk) begin
    ra01 <= o_a01;
    rp2  <= o_m2;
    rp3  <= o_m3;
    if (ost == O_PREP || ost == O_RUN) begin
        case (pcur)
        2'd0: hn0 <= o_dz;
        2'd1: hn1 <= o_dz;
        2'd2: hn2 <= o_dz;
        default: begin
            ho0 <= hn0;
            ho1 <= hn1;
            ho2 <= hn2;
            ho3 <= o_dz;
        end
        endcase
    end
end

//---------------------------------------------------------------------
//  control (asynchronous active-low reset)
//---------------------------------------------------------------------
wire first_issue7 = issue && (sb == 3'd7) && (sidx == 4'd0) && !shd && !sq_r;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ld_cnt    <= 7'd0;
        ld_q      <= 7'd0;
        sb        <= 3'd0;
        sidx      <= 4'd0;
        shd       <= 1'b0;
        sq_r      <= 1'b0;
        sdone     <= 1'b0;
        va <= 1'b0; vb <= 1'b0; vc <= 1'b0; vd <= 1'b0; ve <= 1'b0;
        p7_on     <= 1'b0;
        p7c       <= 4'd0;
        ost       <= O_IDLE;
        pc        <= 2'd0;
        oc        <= 6'd0;
        ov2       <= 1'b0;
        out_valid <= 1'b0;
        out       <= 32'd0;
    end
    else begin
        ld_q <= ld_cnt;
        if (in_valid) ld_cnt <= ld_cnt + 7'd1;

        // pipeline valid bits
        va <= issue;
        vb <= va;
        vc <= vb;
        vd <= vc;
        ve <= vd;

        // tile sequencer
        if (issue) begin
            sq_r <= ~sq_r;
            if (sq_r) begin
                shd <= ~shd;
                if (shd) begin
                    if (sidx == {sb, 1'b0}) begin
                        sidx <= 4'd0;
                        if (sb == 3'd7) sdone <= 1'b1;
                        else            sb <= sb + 3'd1;
                    end
                    else sidx <= sidx + 4'd1;
                end
            end
        end

        // output start trigger
        if (first_issue7) begin
            p7_on <= 1'b1;
            p7c   <= 4'd0;
        end
        else if (p7_on) p7c <= p7c + 4'd1;

        // output engine
        ov2       <= (ost == O_RUN);
        out_valid <= ov2;
        out       <= ov2 ? o_c : 32'd0;

        case (ost)
        O_IDLE: begin
            if (p7_on && p7c == P7_DLY) begin
                ost   <= O_PREP;
                pc    <= 2'd0;
                p7_on <= 1'b0;
            end
        end
        O_PREP: begin
            pc <= pc + 2'd1;
            if (pc == 2'd3) begin
                ost <= O_RUN;
                oc  <= 6'd0;
            end
        end
        O_RUN: begin
            oc <= oc + 6'd1;
            if (oc == 6'd63) ost <= O_FLUSH;
        end
        O_FLUSH: ost <= O_DONE;
        O_DONE: begin                   // clear for the next run
            ost    <= O_IDLE;
            ld_cnt <= 7'd0;
            ld_q   <= 7'd0;
            sb     <= 3'd0;
            sidx   <= 4'd0;
            shd    <= 1'b0;
            sq_r   <= 1'b0;
            sdone  <= 1'b0;
            p7_on  <= 1'b0;
            p7c    <= 4'd0;
            pc     <= 2'd0;
            oc     <= 6'd0;
        end
        default: ost <= O_IDLE;
        endcase
    end
end

`undef FA_MUL
`undef FA_ADD
`undef FA_DIV
`undef FA_DP2
`undef FA_EXP
`undef FA_GT

endmodule
