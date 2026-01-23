from thop import profile
from models.restormer_volterra import RestormerVolterra
model = RestormerVolterra()
params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {params/1e6:.2f} M")
