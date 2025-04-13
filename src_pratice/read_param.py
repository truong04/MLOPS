from metaflow import Flow
run = Flow("FullMLFlow")[2]  # Thay tên flow và run id nếu cần

for var in dir(run.data):
    if not var.startswith('_'):
        print(var)
        val = getattr(run.data, var)
        print(f"--{var} {val}")

