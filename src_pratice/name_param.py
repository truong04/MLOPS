from metaflow import FlowSpec, step, Parameter

class ParamFlow(FlowSpec):
    who = Parameter('who', 
                     help='name param', 
                     default='Metaflow',
                     type=str)

    @step
    def start(self):
        print(f"Xin chào {self.who}")
        self.next(self.end)

    @step
    def end(self):
        print("Kết thúc")

if __name__ == '__main__':
    ParamFlow()