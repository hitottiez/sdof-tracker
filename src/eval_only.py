import os
import sys
import motmetrics as mm
from opts import Opts

sys.path.append("../../FairMOT/src/lib")
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator


# Read opts
opts = Opts()
opts.check_params()

# Prepare
accs = [] # Accuracies

# Open file
os.makedirs(opts.output_motmetrics_dir, exist_ok=True)

for seq_name in opts.seq_names:
    print(seq_name)
    output_track_dir = "{}/tracks".format(opts.output_root_dir)

    # Open file
    output_track_path = "{}/{}.txt".format(output_track_dir, seq_name)

    # Evaluate
    if opts.eval_on:
        evaluator = Evaluator(opts.input_root_dir, seq_name, "mot")
        accs.append(evaluator.eval_file(output_track_path))

# Summarize evaluation
if opts.eval_on:
    #mm.lap.default_solver = 'scipy'
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, opts.seq_names, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(opts.output_motmetrics_dir, 'summary.xlsx'))
