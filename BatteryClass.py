# coding : utf8
'''
Values are in hours unless otherwise stated
'''
class Battery:
    def __init__(self, MaxChargingRate = 2, MaxStorage = 4, ChargingEfficiency = 0.95, 
                 Lifetime = 87600, MaxCycles = 5000, MaxStorageLoss = 0.00001, 
                 Charge = 0, Cycles = 0, UpTime = 0, Bank = 0):
        self.MaxChargingRate = MaxChargingRate
        self.MaxStorage = MaxStorage
        self.ChargingEfficiency = ChargingEfficiency
        self.Lifetime = Lifetime
        self.MaxCycles = MaxCycles
        self.MaxStorageLoss = MaxStorageLoss
        self.Charge = Charge
        self.Cycles = Cycles
        self.UpTime = UpTime
        self.Bank = Bank
    def UpdateBattery(self, ChargingRate):
        self.UpTime += 0.5
        if ChargingRate > self.MaxChargingRate:
            self.PrintAttributes()
            raise Exception('Charging rate must be less than the max charging rate')
        if self.Charge > self.MaxStorage:
            self.PrintAttributes()
            raise Exception('Charge must be less than the maximum capacity')
        if self.Cycles >= self.MaxCycles:
            self.MaxStorage = 0
            self.MaxChargingRate = 0
            self.PrintAttributes()
            raise Exception('Maximum cycles reached, the battery is dead')
        if self.UpTime >= self.Lifetime:
            self.MaxStorage = 0
            self.MaxChargingRate = 0
            self.PrintAttributes()
            raise Exception('Maximum lifetime reached, the battery is dead')
    def Charging(self, ChargingRate, Cost, ChargeTime = 0.5):
        self.UpdateBattery(ChargingRate)
        if self.Charge + self.ChargingEfficiency * ChargingRate * ChargeTime > self.MaxStorage:
            self.PrintAttributes()
            raise Exception('Charge must be less than the maximum capacity')
        if self.Charge + self.ChargingEfficiency * ChargingRate * ChargeTime < 0:
            self.PrintAttributes()
            raise Exception('Charge must be more than the minimum capacity')
        self.Charge += self.ChargingEfficiency * ChargingRate * ChargeTime 
        self.Cycles += abs(self.ChargingEfficiency * ChargingRate * ChargeTime * 0.5)/self.MaxStorage # Half lifetime either going up or down
        self.MaxStorage *= 1 - self.MaxStorageLoss * abs(self.ChargingEfficiency * ChargingRate * ChargeTime * 0.5)
        self.Bank -= Cost * ChargingRate * ChargeTime
    def PrintAttributes(self):
        print(str(self.MaxChargingRate), 'MaxChargingRate \n', str(self.MaxStorage), 'MaxStorage \n',
        str(self.ChargingEfficiency), 'ChargingEfficiency \n', str(self.Lifetime), 'Lifetime \n',
        str(self.MaxCycles), 'MaxCycles \n', str(self.MaxStorageLoss), 'MaxStorageLoss \n',
        str(self.Charge), 'Charge \n', str(self.Cycles), 'Cycles \n',
        str(self.UpTime), 'Uptime \n', str(self.Bank), 'Bank \n')