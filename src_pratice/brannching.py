from metaflow import FlowSpec, step, card

class BranchingFlow(FlowSpec):
    
    @card
    @step
    def start(self):
        print("Bắt đầu flow")
        self.next(self.split)
    
    @card
    @step
    def split(self):
        self.next(self.branch1, self.branch2)
    
    @card
    @step
    def branch1(self):
        print("Đây là nhánh 1")
        self.next(self.join)
    
    @card
    @step
    def branch2(self):
        print("Đây là nhánh 2")
        self.next(self.join)
    
    @card
    @step
    def join(self, inputs):
        print("Kết thúc branching")
        self.next(self.end)
        
    @card
    @step
    def end(self):
        print("Kết thúc flow")

if __name__ == '__main__':
    BranchingFlow()
