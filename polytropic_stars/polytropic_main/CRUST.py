import numpy as np 


class CRUST:
    def __init__(self,P):
        self.P = P
    
    def crust_equation1(self):
        return 103.17338*(1 - np.exp(-self.P/0.38527)) + 7.34979* (1 - np.exp(-self.P/0.01211))+ 0.00873
    
    def crust_equation2(self):
        return 0.00015 + 0.00203 * (1 - np.exp(-self.P * 344827.5)) + 0.10851 * (1 - np.exp(-self.P * 7692.3076))
        
    def crust_equation3(self):
        return 0.0000051 * (1 - np.exp(- self.P * 0.2373 * 1e10) ) + 0.00014 * (1 - np.exp(- self.P * 0.4020 * 1e8))                        
    
    def crust_equation4(self):
        c0 = 31.93753
        c1 = 10.82611 * np.log(self.P) / np.log(10)
        c2 = 1.29312 * (np.log(self.P) / np.log(10))**2
        c3 = 0.08014 * (np.log(self.P) / np.log(10))**3
        c4 = 0.00242 * (np.log(self.P) / np.log(10))**4
        c5 = 0.000028 * (np.log(self.P) / np.log(10))**5
        return 10**(c0 + c1 + c2 +c3 + c4 + c5)
    def equation(self):
        if 9.34375 * 1e-5 <= self.P <= 0.184:
            return self.crust_equation1()
        
        if 4.1725 * 1e-8  <= self.P < 9.34375 * 1e-5:
            return self.crust_equation2()
            
        if 1.44875 * 1e-11 <= self.P < 4.1725 * 1e-8:
            return self.crust_equation3()
            
        if self.P < 1.44875 * 1e-11:
            return self.crust_equation4()


