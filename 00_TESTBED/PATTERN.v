`timescale 1ns/1ps

`define CYCLE_TIME      50.0        // keep identical to "set CYCLE" in 02_SYN/syn.tcl
`define SEED_NUMBER     28825252
`ifndef PATTERN_NUMBER
  `define PATTERN_NUMBER 1000
`endif
`define ERR_TOL         1.0e-7
`define MAX_LATENCY     1000
`define DIRECTED_NUM    13          // corner-case patterns before the random ones
`ifndef GAP_MIN
  `define GAP_MIN       1           // negative edges to wait after the first idle sample (1 -> in_valid at the 2nd edge)
`endif

// Two modes
//   default          : read ../00_TESTBED/input.txt and output.txt, drive the DUT, compare
//   +define+GEN_DATA : generate random / directed data, compute the golden answer with the
//                      DesignWare IP and write input.txt / output.txt (the DUT is not run)
//
// input.txt  (per pattern, 208 words) : 64 lines "K Q V" (raster order, hex), then 16 lines "out_weight"
// output.txt (per pattern,  64 words) : 64 lines of the expected out (raster order, hex)
`ifndef INPUT_FILE
  `define INPUT_FILE    "../00_TESTBED/input.txt"
`endif
`ifndef OUTPUT_FILE
  `define OUTPUT_FILE   "../00_TESTBED/output.txt"
`endif



module PATTERN (
    clk,
    rst_n,
    in_valid,
    K,
    Q,
    V,
    out_weight,
    out_valid,
    out
);

output reg          clk;
output reg          rst_n;
output reg          in_valid;
output reg  [31:0]  K;
output reg  [31:0]  Q;
output reg  [31:0]  V;
output reg  [31:0]  out_weight;

input  wire         out_valid;
input  wire [31:0]  out;

//=============================================================================
// Parameters
//=============================================================================
parameter CYCLE          = `CYCLE_TIME;



parameter inst_sig_width       = 23;
parameter inst_exp_width       = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch_type       = 0;
parameter inst_arch            = 0;

localparam [31:0] SQRT2   = 32'h3FB504F3;   // sqrt(2) given in the lab PDF
localparam [31:0] F_HALF  = 32'h3F000000;   //  0.5
localparam [31:0] F_255   = 32'h437F0000;   //  255.0

//=============================================================================
// Clock
//=============================================================================
always #(CYCLE/2.0) clk = ~clk;
initial clk = 1'b0;

//=============================================================================
// Common variables / helpers
//=============================================================================
integer seed;
integer pat, i, lat, gap;
integer total_lat, max_lat;
real    max_err, cur_err, gold_r, out_r;
reg     chk_en;                      // idle monitor enable (after reset)

// float32 bit pattern -> real
function real f32_to_real;
    input [31:0] b;
    integer ex;
    real    fr, v;
    begin
        ex = b[30:23];
        fr = b[22:0];
        if (ex == 0) v = 0.0;
        else         v = (1.0 + fr / 8388608.0) * (2.0 ** (ex - 127));
        f32_to_real = b[31] ? -v : v;
    end
endfunction

function real fabs;
    input real x;
    begin
        fabs = (x < 0.0) ? -x : x;
    end
endfunction

task drive_idle;
    begin
        in_valid   = 1'b0;
        Q          = 32'bx;
        K          = 32'bx;
        V          = 32'bx;
        out_weight = 32'bx;
    end
endtask


`ifdef GEN_DATA
//#############################################################################
//  GENERATION MODE : data + golden -> input.txt / output.txt
//#############################################################################
reg [31:0] q_mem    [0:`PATTERN_NUMBER*64-1];
reg [31:0] k_mem    [0:`PATTERN_NUMBER*64-1];
reg [31:0] v_mem    [0:`PATTERN_NUMBER*64-1];
reg [31:0] w_mem    [0:`PATTERN_NUMBER*16-1];
reg [31:0] gold_mem [0:`PATTERN_NUMBER*64-1];
integer gb, gwb;                     // base index of the pattern being prepared

function [31:0] neg;
    input [31:0] x;
    begin
        neg = {~x[31], x[30:0]};
    end
endfunction

// round-to-nearest-even
function [31:0] real_to_f32;
    input real x;
    reg [63:0] bb;
    reg        sg;
    integer    e;
    reg [22:0] fr;
    reg [28:0] rem;
    reg [23:0] sm;
    begin
        bb = $realtobits(x);
        sg = bb[63];
        e  = bb[62:52];
        if (e == 0) real_to_f32 = {sg, 31'b0};
        else begin
            e   = e - 1023 + 127;
            fr  = bb[51:29];
            rem = bb[28:0];
            if (rem > 29'h10000000 || (rem == 29'h10000000 && fr[0])) begin
                sm = {1'b0, fr} + 24'd1;
                fr = sm[22:0];
                if (sm[23]) e = e + 1;
            end
            if (e <= 0)        real_to_f32 = {sg, 31'b0};
            else if (e >= 255) real_to_f32 = {sg, 8'hFF, 23'b0};
            else               real_to_f32 = {sg, e[7:0], fr};
        end
    end
endfunction

function real urand;                 // uniform in [0,1)
    input dummy;
    reg [31:0] r;
    begin
        r = $random(seed);
        urand = r / 4294967296.0;
    end
endfunction

// mode 0 : uniform magnitude, 1 : log-uniform (zero allowed when lo==0), 2 : mostly range limits
// the sign is random; the result is a float32 bit pattern
function [31:0] rand_f32;
    input real    lo;
    input real    hi;
    input integer mode;
    real    m, lo_l;
    integer pick;
    reg     sg;
    reg [31:0] f;
    begin
        lo_l = (lo > 1.0e-3) ? lo : 1.0e-3;
        case (mode)
            1: begin
                m = $exp($ln(lo_l) + ($ln(hi) - $ln(lo_l)) * urand(0));
                if (lo == 0.0 && urand(0) < 0.05) m = 0.0;
            end
            2: begin
                pick = {$random(seed)} % 3;
                m = (pick == 0) ? lo : (pick == 1) ? hi : (lo + (hi - lo) * urand(0));
            end
            default: m = lo + (hi - lo) * urand(0);
        endcase
        sg = $random(seed);
        f  = real_to_f32(m);
        if (m == 0.0) f = 32'd0;
        else          f[31] = sg;
        rand_f32 = f;
    end
endfunction

task rnd_q; begin for (i = 0; i < 64; i = i + 1) q_mem[gb + i] = rand_f32(0.0, 0.5,   0); end endtask
task rnd_k; begin for (i = 0; i < 64; i = i + 1) k_mem[gb + i] = rand_f32(0.5, 255.0, 0); end endtask
task rnd_v; begin for (i = 0; i < 64; i = i + 1) v_mem[gb + i] = rand_f32(0.0, 0.5,   0); end endtask
task rnd_w; begin for (i = 0; i < 16; i = i + 1) w_mem[gwb + i] = rand_f32(0.0, 0.5,  0); end endtask
task fill_q; input [31:0] c; begin for (i = 0; i < 64; i = i + 1) q_mem[gb + i] = c; end endtask
task fill_k; input [31:0] c; begin for (i = 0; i < 64; i = i + 1) k_mem[gb + i] = c; end endtask
task fill_v; input [31:0] c; begin for (i = 0; i < 64; i = i + 1) v_mem[gb + i] = c; end endtask
task fill_w; input [31:0] c; begin for (i = 0; i < 16; i = i + 1) w_mem[gwb + i] = c; end endtask

// corner cases: exp() range, running max tracking, zero data
task directed_task;
    input integer p;
    integer t, d;
    begin
        case (p)
            0: begin fill_q(F_HALF); fill_k(F_255);         rnd_v;         rnd_w;         end // all scores max positive
            1: begin fill_q(F_HALF); fill_k(neg(F_255));    rnd_v;         rnd_w;         end // all scores max negative
            2: begin fill_q(32'd0);  rnd_k;                 rnd_v;         rnd_w;         end // all scores 0
            3: begin rnd_q;          rnd_k;                 fill_v(32'd0); rnd_w;         end // V = 0
            4: begin rnd_q;          rnd_k;                 rnd_v;         fill_w(32'd0); end // W = 0
            5: begin rnd_q;                                                                    // smallest keys
                     for (i = 0; i < 64; i = i + 1) k_mem[gb + i] = {$random(seed)} % 2 ? neg(F_HALF) : F_HALF;
                     rnd_v; rnd_w; end
            12: begin fill_q(F_HALF);                                                          // every score negative
                     for (i = 0; i < 64; i = i + 1) k_mem[gb + i] = neg(rand_f32(0.5, 255.0, 0) & 32'h7FFFFFFF);
                     rnd_v; rnd_w; end
            default: begin                                                                     // one dominant key token
                case (p)
                    6:  t = 0;
                    7:  t = 1;
                    8:  t = 7;
                    9:  t = 8;
                    10: t = 14;
                    default: t = 15;
                endcase
                fill_q(F_HALF);
                fill_k(F_HALF);
                for (d = 0; d < 4; d = d + 1) k_mem[gb + t*4 + d] = F_255;
                rnd_v; rnd_w;
            end
        endcase
    end
endtask

task gen_task;
    input integer p;
    integer kind, mode;
    begin
        gb  = p * 64;
        gwb = p * 16;
        if (p < `DIRECTED_NUM) directed_task(p);
        else begin
            kind = (p - `DIRECTED_NUM) % 10;
            mode = (kind < 6) ? 0 : (kind < 8) ? 1 : 2;
            for (i = 0; i < 64; i = i + 1) begin
                q_mem[gb + i] = rand_f32(0.0, 0.5,   mode);
                k_mem[gb + i] = rand_f32(0.5, 255.0, mode);
                v_mem[gb + i] = rand_f32(0.0, 0.5,   mode);
            end
            for (i = 0; i < 16; i = i + 1) w_mem[gwb + i] = rand_f32(0.0, 0.5, mode);
        end
    end
endtask

//-----------------------------------------------------------------------------
// Golden model: tiled online softmax built from the same DesignWare IP as F_ATTN
//-----------------------------------------------------------------------------
reg  [31:0] gm_a, gm_b;  wire [31:0] gm_z;
reg  [31:0] ga_a, ga_b;  wire [31:0] ga_z;
reg  [31:0] gd_a, gd_b;  wire [31:0] gd_z;
reg  [31:0] ge_a;        wire [31:0] ge_z;
reg  [31:0] gc_a, gc_b;  wire        gc_gt;

DW_fp_mult #(inst_sig_width, inst_exp_width, inst_ieee_compliance)
    G_MULT (.a(gm_a), .b(gm_b), .rnd(3'b000), .z(gm_z), .status());
DW_fp_add  #(inst_sig_width, inst_exp_width, inst_ieee_compliance)
    G_ADD  (.a(ga_a), .b(ga_b), .rnd(3'b000), .z(ga_z), .status());
DW_fp_div  #(inst_sig_width, inst_exp_width, inst_ieee_compliance)
    G_DIV  (.a(gd_a), .b(gd_b), .rnd(3'b000), .z(gd_z), .status());
DW_fp_exp  #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_arch)
    G_EXP  (.a(ge_a), .z(ge_z), .status());
DW_fp_cmp  #(inst_sig_width, inst_exp_width, inst_ieee_compliance)
    G_CMP  (.a(gc_a), .b(gc_b), .zctr(1'b0), .aeqb(), .altb(), .agtb(gc_gt),
            .unordered(), .z0(), .z1(), .status0(), .status1());
`ifdef GOLD_QK_DP2
reg  [31:0] gp_a, gp_b, gp_c, gp_d;  wire [31:0] gp_z;
DW_fp_dp2  #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_arch_type)
    G_DP2  (.a(gp_a), .b(gp_b), .c(gp_c), .d(gp_d), .rnd(3'b000), .z(gp_z), .status());
task fdp2; input [31:0] a, b, c, d; begin gp_a = a; gp_b = b; gp_c = c; gp_d = d; #1; t_z = gp_z; end endtask
`endif

reg [31:0] t_z;
reg        t_g;

task fmul; input [31:0] a, b; begin gm_a = a; gm_b = b; #1; t_z = gm_z; end endtask
task fadd; input [31:0] a, b; begin ga_a = a; ga_b = b; #1; t_z = ga_z; end endtask
task fdiv; input [31:0] a, b; begin gd_a = a; gd_b = b; #1; t_z = gd_z; end endtask
task fexp; input [31:0] a;    begin ge_a = a;           #1; t_z = ge_z; end endtask
task fgt;  input [31:0] a, b; begin gc_a = a; gc_b = b; #1; t_g = gc_gt; end endtask

reg [31:0] gs   [0:3];       // scaled scores of the current tile
reg [31:0] gp   [0:3];       // exp(s - m_new)
reg [31:0] gacc [0:3];       // O accumulator
reg [31:0] gl   [0:1];       // softmax denominator
reg [31:0] gmx  [0:1];       // running max
reg [31:0] gh   [0:63];      // head outputs, 16 tokens x 4 dims
reg [31:0] x_q0, x_q1, x_k0, x_k1, x_v0, x_v1;
reg [31:0] x_a, x_b, x_c;
reg [31:0] x_rmax, x_mnew, x_alpha, x_psum, x_pv;

task golden_task;
    input integer p;
    integer hh, bi, bj, r, c, d, tk;
    begin
        gb  = p * 64;
        gwb = p * 16;

        for (hh = 0; hh < 2; hh = hh + 1) begin
            for (bi = 0; bi < 8; bi = bi + 1) begin
                for (r = 0; r < 2; r = r + 1) begin
                    gl[r] = 32'd0;
                    gmx[r] = 32'd0;
                end
                for (d = 0; d < 4; d = d + 1) gacc[d] = 32'd0;

                for (bj = 0; bj < 8; bj = bj + 1) begin
                    // S = (q0*k0 + q1*k1) / sqrt2
                    for (r = 0; r < 2; r = r + 1) begin
                        for (c = 0; c < 2; c = c + 1) begin
                            x_q0 = q_mem[gb + (2*bi + r)*4 + 2*hh];
                            x_q1 = q_mem[gb + (2*bi + r)*4 + 2*hh + 1];
                            x_k0 = k_mem[gb + (2*bj + c)*4 + 2*hh];
                            x_k1 = k_mem[gb + (2*bj + c)*4 + 2*hh + 1];
`ifdef GOLD_QK_DP2
                            fdp2(x_q0, x_k0, x_q1, x_k1); x_c = t_z;     // one rounding
`else
                            fmul(x_q0, x_k0); x_a = t_z;
                            fmul(x_q1, x_k1); x_b = t_z;
                            fadd(x_a, x_b);   x_c = t_z;
`endif
                            fdiv(x_c, SQRT2); gs[r*2 + c] = t_z;
                        end
                    end

                    // online softmax update, row by row
                    for (r = 0; r < 2; r = r + 1) begin
                        fgt(gs[r*2 + 1], gs[r*2]);
                        x_rmax = t_g ? gs[r*2 + 1] : gs[r*2];
                        if (bj == 0) x_mnew = x_rmax;
                        else begin
                            fgt(x_rmax, gmx[r]);
                            x_mnew = t_g ? x_rmax : gmx[r];
                        end

                        if (bj == 0) x_alpha = 32'd0;           // exp(-inf) = 0
                        else begin
                            fadd(gmx[r], neg(x_mnew));
                            fexp(t_z);
                            x_alpha = t_z;
                        end

                        for (c = 0; c < 2; c = c + 1) begin
                            fadd(gs[r*2 + c], neg(x_mnew));
                            fexp(t_z);
                            gp[r*2 + c] = t_z;
                        end

                        fadd(gp[r*2], gp[r*2 + 1]); x_psum = t_z;
                        fmul(x_alpha, gl[r]);       x_a = t_z;
                        fadd(x_a, x_psum);          gl[r] = t_z;

                        for (d = 0; d < 2; d = d + 1) begin
                            x_v0 = v_mem[gb + (2*bj)*4     + 2*hh + d];
                            x_v1 = v_mem[gb + (2*bj + 1)*4 + 2*hh + d];
                            fmul(gp[r*2],     x_v0); x_a = t_z;
                            fmul(gp[r*2 + 1], x_v1); x_b = t_z;
                            fadd(x_a, x_b);          x_pv = t_z;
                            fmul(x_alpha, gacc[r*2 + d]); x_c = t_z;
                            fadd(x_c, x_pv);              gacc[r*2 + d] = t_z;
                        end
                        gmx[r] = x_mnew;
                    end
                end

                // head output = O / l
                for (r = 0; r < 2; r = r + 1) begin
                    for (d = 0; d < 2; d = d + 1) begin
                        fdiv(gacc[r*2 + d], gl[r]);
                        gh[(2*bi + r)*4 + 2*hh + d] = t_z;
                    end
                end
            end
        end

        // concat x Out_weight^T : ((h0*w0 + h1*w1) + h2*w2) + h3*w3
        for (tk = 0; tk < 16; tk = tk + 1) begin
            for (c = 0; c < 4; c = c + 1) begin
`ifdef GOLD_WT
                fmul(gh[tk*4 + 0], w_mem[gwb + 0*4 + c]); x_a = t_z;
                fmul(gh[tk*4 + 1], w_mem[gwb + 1*4 + c]); x_b = t_z;
                fadd(x_a, x_b);                           x_c = t_z;
                fmul(gh[tk*4 + 2], w_mem[gwb + 2*4 + c]); x_a = t_z;
                fadd(x_c, x_a);                           x_c = t_z;
                fmul(gh[tk*4 + 3], w_mem[gwb + 3*4 + c]); x_b = t_z;
`else
                fmul(gh[tk*4 + 0], w_mem[gwb + c*4 + 0]); x_a = t_z;
                fmul(gh[tk*4 + 1], w_mem[gwb + c*4 + 1]); x_b = t_z;
                fadd(x_a, x_b);                           x_c = t_z;
                fmul(gh[tk*4 + 2], w_mem[gwb + c*4 + 2]); x_a = t_z;
                fadd(x_c, x_a);                           x_c = t_z;
                fmul(gh[tk*4 + 3], w_mem[gwb + c*4 + 3]); x_b = t_z;
`endif
                fadd(x_c, x_b);
                gold_mem[gb + tk*4 + c] = t_z;
            end
        end
    end
endtask

integer fin, fout, pp;
initial begin
    force clk = 1'b0;
    rst_n = 1'b1;
    drive_idle;
    seed = `SEED_NUMBER;

    for (pat = 0; pat < `PATTERN_NUMBER; pat = pat + 1) begin
        gen_task(pat);
        golden_task(pat);
        if ((pat % 100) == 99) $display("  [PATTERN] %0d / %0d patterns prepared", pat + 1, `PATTERN_NUMBER);
    end

    fin  = $fopen(`INPUT_FILE,  "w");
    fout = $fopen(`OUTPUT_FILE, "w");
    for (pp = 0; pp < `PATTERN_NUMBER; pp = pp + 1) begin
        $fdisplay(fin,  "// pattern %0d", pp);
        for (i = 0; i < 64; i = i + 1)
            $fdisplay(fin, "%h %h %h", k_mem[pp*64 + i], q_mem[pp*64 + i], v_mem[pp*64 + i]);
        for (i = 0; i < 16; i = i + 1)
            $fdisplay(fin, "%h", w_mem[pp*16 + i]);
        $fdisplay(fout, "// pattern %0d", pp);
        for (i = 0; i < 64; i = i + 1)
            $fdisplay(fout, "%h", gold_mem[pp*64 + i]);
    end
    $fclose(fin);
    $fclose(fout);
    $display("--------------------------------------------------------------");
    $display("  %0d patterns written to", `PATTERN_NUMBER);
    $display("    %0s", `INPUT_FILE);
    $display("    %0s", `OUTPUT_FILE);
    $display("--------------------------------------------------------------");
    $finish;
end

`else
//#############################################################################
//  NORMAL MODE : read input.txt / output.txt, drive the DUT and compare
//#############################################################################
reg [31:0] in_mem  [0:`PATTERN_NUMBER*208-1];     // per pattern: 64 x (K,Q,V), then 16 x out_weight
reg [31:0] out_mem [0:`PATTERN_NUMBER*64-1];      // per pattern: 64 expected outputs

//-----------------------------------------------------------------------------
// Failure / success
//-----------------------------------------------------------------------------
task fail_task;
    input [8*60-1:0] msg;
    begin
        $display("--------------------------------------------------------------");
        $display("  FAIL  pattern %0d  : %0s", pat, msg);
        $display("  time  %0t", $time);
        $display("--------------------------------------------------------------");
        $finish;
    end
endtask

task pass_task;
    begin
        $display("--------------------------------------------------------------");
        $display("  Congratulations!  all %0d patterns passed", `PATTERN_NUMBER);
        $display("  max |out - golden| = %e   (tolerance %e)", max_err, `ERR_TOL);
        $display("  latency  max = %0d cycles, average = %0d cycles", max_lat, total_lat / `PATTERN_NUMBER);
        $display("--------------------------------------------------------------");
        $finish;
    end
endtask

//-----------------------------------------------------------------------------
// Output idle monitor: out must be 0 while out_valid is low
//-----------------------------------------------------------------------------
always @(negedge clk) begin
    if (chk_en) begin
        if (^{out_valid} === 1'bx)
            fail_task("out_valid is X");
        else if (out_valid === 1'b0 && out !== 32'd0)
            fail_task("out is not 0 while out_valid is low");
    end
end

//-----------------------------------------------------------------------------
// Tasks
//-----------------------------------------------------------------------------
// asynchronous reset: asserted / released while the clock is held low
task reset_task;
    begin
        rst_n = 1'b1;
        drive_idle;
        chk_en = 1'b0;
        force clk = 1'b0;
        #(CYCLE);
        rst_n = 1'b0;
        #(CYCLE * 2.0);
        if (out_valid !== 1'b0 || out !== 32'd0)
            fail_task("out_valid / out are not 0 after reset");
        #(CYCLE);
        rst_n = 1'b1;
        #(CYCLE);
        release clk;
        chk_en = 1'b1;
    end
endtask

// 64 beats of K/Q/V, out_weight only in the first 16.  Called at a negative edge.
task input_task;
    begin
        for (i = 0; i < 64; i = i + 1) begin
            if (out_valid !== 1'b0) fail_task("out_valid overlaps with in_valid");
            in_valid = 1'b1;
            K = in_mem[pat*208 + i*3];
            Q = in_mem[pat*208 + i*3 + 1];
            V = in_mem[pat*208 + i*3 + 2];
            out_weight = (i < 16) ? in_mem[pat*208 + 192 + i] : 32'bx;
            @(negedge clk);
        end
        drive_idle;                 // in_valid falls at this negative edge
    end
endtask

task wait_out_task;
    begin
        lat = 0;
        while (out_valid !== 1'b1) begin
            if (out_valid !== 1'b0) fail_task("out_valid is X");
            @(negedge clk);
            lat = lat + 1;
            if (lat > `MAX_LATENCY) fail_task("latency exceeds 1000 cycles");
        end
        total_lat = total_lat + lat;
        if (lat > max_lat) max_lat = lat;
    end
endtask

task check_out_task;
    begin
        for (i = 0; i < 64; i = i + 1) begin
            if (out_valid !== 1'b1)  fail_task("out_valid is not high for 64 cycles");
            if (^out === 1'bx)       fail_task("out contains X");
            if (out[30:23] == 8'hff) fail_task("out is Inf / NaN");
            out_r  = f32_to_real(out);
            gold_r = f32_to_real(out_mem[pat*64 + i]);
            cur_err = fabs(out_r - gold_r);
            if (cur_err > max_err) max_err = cur_err;
            if (cur_err >= `ERR_TOL) begin
                $display("  output %0d : out = %h (%e)   golden = %h (%e)   err = %e",
                         i, out, out_r, out_mem[pat*64 + i], gold_r, cur_err);
                fail_task("output error is larger than the tolerance");
            end
            @(negedge clk);
        end
        // 65th sample: out_valid must have dropped and out must be 0
        if (out_valid !== 1'b0) fail_task("out_valid is high for more than 64 cycles");
        if (out !== 32'd0)      fail_task("out is not 0 after out_valid fell");
    end
endtask

//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------
initial begin
    force clk = 1'b0;
    rst_n  = 1'b1;
    chk_en = 1'b0;
    drive_idle;
    pat = 0;

    $readmemh(`INPUT_FILE,  in_mem);
    $readmemh(`OUTPUT_FILE, out_mem);
    if (^in_mem[`PATTERN_NUMBER*208 - 1] === 1'bx || ^out_mem[`PATTERN_NUMBER*64 - 1] === 1'bx) begin
        $display("--------------------------------------------------------------");
        $display("  FAIL  input.txt / output.txt are missing or hold fewer than %0d patterns", `PATTERN_NUMBER);
        $display("        generate them once with +define+GEN_DATA");
        $display("--------------------------------------------------------------");
        $finish;
    end

    seed = `SEED_NUMBER;
    total_lat = 0;
    max_lat = 0;
    max_err = 0.0;

    reset_task;
    @(negedge clk);

    for (pat = 0; pat < `PATTERN_NUMBER; pat = pat + 1) begin
        // spec Fig3: in_valid rises at the 2nd ~ 4th negative edge after out_valid fell.
        // check_out_task ends at the 1st negative edge where out_valid is low, so wait 1 ~ 3 more.
        gap = `GAP_MIN + ({$random(seed)} % 3);
        repeat (gap) @(negedge clk);
        input_task;
        wait_out_task;
        check_out_task;
        if ((pat % 50) == 49) $display("  [PATTERN] %0d / %0d patterns checked, max err so far %e", pat + 1, `PATTERN_NUMBER, max_err);
    end
    pass_task;
end

`endif

endmodule
