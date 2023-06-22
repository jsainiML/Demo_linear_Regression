# Linear Regression base computation model


class lrmodel(nn.Module):
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.rand(1, requires_grad=True))   
    self.bias = nn.Parameter(torch.rand(1,requires_grad=True))
    
  # forward method to define computation in model
  def forward(self, x: torch.tensor) -> torch.tensor:                       # x is the imput data 
    return self.weights*x + self.bias                                       # Linear regression formula
 
 
torch.manual_seed(99)                                                       # Creating random seed so that our parameter have some common randomness (easier understanding).
model_0 = lrmodel()                                                         # Loading model to a instance to later train and test







_____________________________________________________________________________________________________________________________________________________________________________________________________________


Below architecture v.2 is simpler version where we defined parameters using Pytorches 'nn.linear()' module with same endgoal.


class lrmodel1(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features=1, out_features=1) # nn.linear created 2 parameters automatically 
    
  # forward method to define computation in model
  def forward(self, x: torch.tensor) -> torch.tensor:  # x is the imput data 
    return self.linear_layer(x)  # this is very similar to our above event of linear regression formula
    
torch.manual_seed(99)                                                    
model_1 = lrmodel1()                                                         

