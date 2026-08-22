// ============================================================================
// flexbe_sequencer.sv -- the microcoded command processor of Sec. 4.
//
// Fetches 32-byte descriptors in program order, decodes each into a resource
// class, and dispatches into one of three lanes (compute / DMA-in / DMA-out).
// Each lane holds a small queue and a countdown timer for its head; a command
// starts only once the tag it depends on has retired (a done-bit scoreboard).
// Completion is by memory: the retiring command's tag is emitted on complete_*.
//
// This is control-plane RTL.  Each dispatched command carries a cycle budget
// (from the Eq. 10 datapath model) that its lane counts down, reproducing the
// DMA/compute overlap the Python sequencer measures.  Wiring the compute lane
// to flexbe_engine.start/done and the DMA lanes to AXI-DMA is the integration
// step.
// ============================================================================
module flexbe_sequencer
  import flexbe_pkg::*;
  import flexbe_desc::*;
#(
  parameter int RING_SLOTS = 16,
  parameter int TAG_W      = 8
)(
  input  logic          clk,
  input  logic          rst_n,

  // descriptor fetch stream (program order), with accompanying cycle budget
  input  logic          fetch_valid,
  output logic          fetch_ready,
  input  logic [255:0]  fetch_desc,
  input  logic [31:0]   fetch_budget,

  // completion notification
  output logic          complete_valid,
  output logic [TAG_W-1:0] complete_tag,

  output logic          idle
);

  localparam int NLANE = 3;
  localparam int NTAG  = (1 << TAG_W);

  // ---- scoreboard ---------------------------------------------------------
  logic [NTAG-1:0] done_q;

  // ---- decode + route -----------------------------------------------------
  descriptor_t d;
  logic [1:0]  rc;
  assign d  = decode(fetch_desc);
  assign rc = res_class(d.opcode);

  logic [TAG_W-1:0] dep_tag;
  logic             has_dep;
  assign dep_tag = d.tag[TAG_W-1:0] - 1'b1;
  assign has_dep = (d.tag != 32'd0);

  // ---- per-lane queues (simple circular buffers) --------------------------
  // Each entry: {tag, cycles, dep_tag, has_dep}
  localparam int EW = TAG_W + 32 + TAG_W + 1;

  logic [EW-1:0]        q_mem   [NLANE][RING_SLOTS];
  logic [$clog2(RING_SLOTS):0] q_head [NLANE];
  logic [$clog2(RING_SLOTS):0] q_tail [NLANE];

  function automatic logic q_empty(input int l);
    return (q_head[l] == q_tail[l]);
  endfunction
  function automatic logic q_full(input int l);
    return ((q_tail[l] - q_head[l]) == RING_SLOTS[$clog2(RING_SLOTS):0]);
  endfunction

  // lane head decode
  logic [TAG_W-1:0] head_tag  [NLANE];
  logic [31:0]      head_cyc  [NLANE];
  logic [TAG_W-1:0] head_dep  [NLANE];
  logic             head_hdep [NLANE];
  always_comb begin
    for (int l = 0; l < NLANE; l++) begin
      logic [EW-1:0] e;
      e = q_mem[l][q_head[l][$clog2(RING_SLOTS)-1:0]];
      {head_tag[l], head_cyc[l], head_dep[l], head_hdep[l]} = e;
    end
  end

  // ---- lane run state -----------------------------------------------------
  logic [31:0]      run_cnt  [NLANE];
  logic             running  [NLANE];
  logic [TAG_W-1:0] cur_tag  [NLANE];
  logic             retire   [NLANE];

  // enqueue accept: fetch goes to the lane of its class if not full
  logic fetch_accept;
  assign fetch_accept = fetch_valid && !q_full(rc);
  assign fetch_ready  = fetch_accept;

  integer li;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done_q <= '0;
      for (li = 0; li < NLANE; li++) begin
        q_head[li]  <= '0;
        q_tail[li]  <= '0;
        running[li] <= 1'b0;
        run_cnt[li] <= '0;
      end
    end else begin
      // default
      for (li = 0; li < NLANE; li++) retire[li] <= 1'b0;

      // enqueue
      if (fetch_accept) begin
        q_mem[rc][q_tail[rc][$clog2(RING_SLOTS)-1:0]]
          <= {d.tag[TAG_W-1:0], fetch_budget, dep_tag, has_dep};
        q_tail[rc] <= q_tail[rc] + 1'b1;
      end

      // per-lane execution
      for (li = 0; li < NLANE; li++) begin
        if (running[li]) begin
          if (run_cnt[li] == 32'd1) begin
            running[li] <= 1'b0;
            done_q[cur_tag[li]] <= 1'b1;
            retire[li] <= 1'b1;
          end else begin
            run_cnt[li] <= run_cnt[li] - 1'b1;
          end
        end else if (!q_empty(li)) begin
          // may start head if its dependency is satisfied
          if (!head_hdep[li] || done_q[head_dep[li]]) begin
            cur_tag[li] <= head_tag[li];
            q_head[li]  <= q_head[li] + 1'b1;
            if (head_cyc[li] == 32'd0) begin
              done_q[head_tag[li]] <= 1'b1;
              retire[li] <= 1'b1;
            end else begin
              run_cnt[li] <= head_cyc[li];
              running[li] <= 1'b1;
            end
          end
        end
      end
    end
  end

  // ---- completion (priority-encode the retiring lane) ---------------------
  always_comb begin
    complete_valid = 1'b0;
    complete_tag   = '0;
    for (int l = 0; l < NLANE; l++)
      if (retire[l]) begin
        complete_valid = 1'b1;
        complete_tag   = cur_tag[l];
      end
  end

  logic any_busy;
  always_comb begin
    any_busy = 1'b0;
    for (int l = 0; l < NLANE; l++)
      any_busy = any_busy || running[l] || !q_empty(l);
  end
  assign idle = !fetch_valid && !any_busy;

endmodule : flexbe_sequencer
