//############################################################################
//   ICLAB 2026 Fall  Lab04 : Simplified Flash Attention (F_ATTN)
//   File Name   : PATTERN.v
//   Module Name : PATTERN
//
//   Self-written pattern.  Input / golden data come from gen_pattern.py
//   (run it once:  python3 gen_pattern.py --n 1000  -> 00_TESTBED/DAT/*.dat)
//
//   Checks:
//     - asynchronous reset: out_valid / out are 0 after reset (clock is held low)
//     - in_valid is high for 64 cycles, inputs change at the NEGATIVE clock edge
//       (K,Q,V : 64 words, out_weight : first 16 words only, X afterwards / when in_valid=0)
//     - latency (falling in_valid -> rising out_valid) <= 1000 cycles
//     - out_valid is high for exactly 64 cycles, never overlaps in_valid
//     - out == 0 whenever out_valid == 0
//     - |out - golden| < ERR_TOL for all 64 outputs (out must be finite, no X)
//     - next input comes 2~4 negative edges after out_valid falls
//############################################################################
`define CYCLE_TIME      50.0      // keep identical to "set CYCLE" in 02_SYN/syn.tcl
`define SEED_NUMBER     28825252
`define PATTERN_NUMBER  1000      // must not exceed --n given to gen_pattern.py
`define ERR_TOL         1.0e-7
`define MAX_LATENCY     1000

module PATTERN(
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

output reg        clk, rst_n, in_valid;
output reg [31:0] Q, K, V, out_weight;
input             out_valid;
input      [31:0] out;

//---------------------------------------------------------------------
//  clock
//---------------------------------------------------------------------
parameter CYCLE = `CYCLE_TIME;
always #(CYCLE/2.0) clk = ~clk;
initial clk = 1'b0;

//---------------------------------------------------------------------
//  data
//---------------------------------------------------------------------
reg [31:0] q_mem    [0:`PATTERN_NUMBER*64-1];
reg [31:0] k_mem    [0:`PATTERN_NUMBER*64-1];
reg [31:0] v_mem    [0:`PATTERN_NUMBER*64-1];
reg [31:0] w_mem    [0:`PATTERN_NUMBER*16-1];
reg [63:0] gold_mem [0:`PATTERN_NUMBER*64-1];   // float64 pass/fail reference
reg [63:0] ref_mem  [0:`PATTERN_NUMBER*64-1];   // float64 truth, info only

integer seed;
integer pat, i, lat, gap;
integer total_lat, max_lat;
real    max_err, max_err_ref;
real    cur_err, cur_err_ref;
real    gold_r, ref_r, out_r;
reg     chk_en;              // output-idle monitor enable (after reset)

//---------------------------------------------------------------------
//  float32 bit pattern -> real  (plain Verilog-2001, no $bitstoshortreal)
//---------------------------------------------------------------------
function real f32_to_real;
    input [31:0] b;
    integer ex;
    real    fr, v;
    begin
        ex = b[30:23];
        fr = b[22:0];
        if (ex == 0) v = (fr / 8388608.0) * (2.0 ** (-126));
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

//---------------------------------------------------------------------
//  failure / success
//---------------------------------------------------------------------
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
        $display("  max |out - golden|     = %e   (tolerance %e)", max_err, `ERR_TOL);
        $display("  max |out - float64 ref|= %e   (information only)", max_err_ref);
        $display("  latency  max = %0d cycles, average = %0d cycles", max_lat, total_lat / `PATTERN_NUMBER);
        $display("--------------------------------------------------------------");
        $finish;
    end
endtask

//---------------------------------------------------------------------
//  output idle monitor: out must be 0 while out_valid is low
//---------------------------------------------------------------------
always @(negedge clk) begin
    if (chk_en) begin
        if (^{out_valid} === 1'bx)
            fail_task("out_valid is X");
        else if (out_valid === 1'b0 && out !== 32'd0)
            fail_task("out is not 0 while out_valid is low");
    end
end

//---------------------------------------------------------------------
//  tasks
//---------------------------------------------------------------------
task drive_idle;
    begin
        in_valid   = 1'b0;
        Q          = 32'bx;
        K          = 32'bx;
        V          = 32'bx;
        out_weight = 32'bx;
    end
endtask

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
            Q = q_mem[pat*64 + i];
            K = k_mem[pat*64 + i];
            V = v_mem[pat*64 + i];
            out_weight = (i < 16) ? w_mem[pat*16 + i] : 32'bx;
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
            if (out_valid !== 1'b1) fail_task("out_valid is not high for 64 cycles");
            if (^out === 1'bx)      fail_task("out contains X");
            if (out[30:23] == 8'hff) fail_task("out is Inf / NaN");
            out_r  = f32_to_real(out);
            gold_r = $bitstoreal(gold_mem[pat*64 + i]);
            ref_r  = $bitstoreal(ref_mem [pat*64 + i]);
            cur_err     = fabs(out_r - gold_r);
            cur_err_ref = fabs(out_r - ref_r);
            if (cur_err     > max_err)     max_err     = cur_err;
            if (cur_err_ref > max_err_ref) max_err_ref = cur_err_ref;
            if (cur_err >= `ERR_TOL) begin
                $display("  output %0d : out = %h (%e)   golden = %e   err = %e",
                         i, out, out_r, gold_r, cur_err);
                fail_task("output error is larger than the tolerance");
            end
            @(negedge clk);
        end
        // 65th sample: out_valid must have dropped and out must be 0
        if (out_valid !== 1'b0) fail_task("out_valid is high for more than 64 cycles");
        if (out !== 32'd0)      fail_task("out is not 0 after out_valid fell");
    end
endtask

//---------------------------------------------------------------------
//  main
//---------------------------------------------------------------------
initial begin
    $readmemh("../00_TESTBED/DAT/Q.dat",    q_mem);
    $readmemh("../00_TESTBED/DAT/K.dat",    k_mem);
    $readmemh("../00_TESTBED/DAT/V.dat",    v_mem);
    $readmemh("../00_TESTBED/DAT/W.dat",    w_mem);
    $readmemh("../00_TESTBED/DAT/GOLD.dat", gold_mem);
    $readmemh("../00_TESTBED/DAT/REF.dat",  ref_mem);

    seed = `SEED_NUMBER;
    total_lat = 0;
    max_lat = 0;
    max_err = 0.0;
    max_err_ref = 0.0;
    pat = 0;

    reset_task;
    @(negedge clk);

    for (pat = 0; pat < `PATTERN_NUMBER; pat = pat + 1) begin
        gap = 2 + ({$random(seed)} % 3);        // 2 ~ 4 negative edges
        repeat (gap) @(negedge clk);
        input_task;
        wait_out_task;
        check_out_task;
    end
    pass_task;
end

endmodule
