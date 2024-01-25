class SIM:
    def __init__(self,a):
        self.a = a

class SIM2D(SIM):
    def __init__(self,a,b):
        super().__init__(a)
        self.b = b

        print('self.b',self.b)
        self.c = dict(zip(self.b,self.b))
        print('self.c',self.c)

class SIM2D_TotallyConstrained(SIM2D):
    def __init__(self,a,b):
        super().__init__(a,b)

if __name__ == '__main__':
    sim2dtc = SIM2D_TotallyConstrained(
        a = 'hi',
        b= [1,2,3]
    )
