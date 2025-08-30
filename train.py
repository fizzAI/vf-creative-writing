import verifiers as vf

# 1. Create environment
env = vf.load_environment("creative-writing")

# 2. Load model
model, tokenizer = vf.get_model_and_tokenizer("google/gemma-3-270m-it")

# 3. Configure training  
args = vf.grpo_defaults(run_name="g3-270m-creative-writing")

# Logging configuration
args.logging_steps = 1
args.log_completions = True
args.report_to = "wandb"  # or "none" to disable
args.num_completions_to_print = 5  # Sample size to log

# 4. Train
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)
trainer.train()

