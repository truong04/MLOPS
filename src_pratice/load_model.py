from metaflow import Flow

run = Flow("IrisFlow").latest_run
print("Run ID:", run.id)
print("Data path:", run.pathspec)
for step in run:
    print(f"Step: {step.id}")
    for task in step:
        print(f" - Task path: {task.pathspec}")
        print(f" - Artifacts: {list(task.data._artifacts.keys())}")
