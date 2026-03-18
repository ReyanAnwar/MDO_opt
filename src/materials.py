# list of materials

class pla:
    def __init__(self):
        self.name = self.__class__.__name__
        self.price = 12 # USD/kg
        self.E = 3 # GPa
        self.rho = 1.24e-3 # kg/m^3
        self.sigma_allow = 83 # MPa
        self.nu = 0.35

class xps:
    def __init__(self):
        self.name = self.__class__.__name__
        self.price = 8 # USD/kg
        self.E = 3.25 # GPa
        self.rho = 0.048e-3 # kg/m^3
        self.sigma_allow = 0.7 # MPa
        self.nu = 0.3

class aluminum_6061:
    def __init__(self):
        self.name = self.__class__.__name__
        self.price = 4 # USD/kg
        self.E = 69 # GPa
        self.rho = 2.7e-3 # kg/m^3
        self.sigma_allow = 276 # MPa
        self.nu = 0.33

class carbon:
    def __init__(self):
        self.name = self.__class__.__name__
        self.price = 12 # USD/kg
        self.E = 110 # GPa
        self.rho = 1.7e-3 # kg/m^3
        self.sigma_allow = 1379 # MPa
        self.nu = 0.25
