import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random

class FeatureEncoder(nn.Module):
  def __init__(self, obs_dim, output_dim):
    super(FeatureEncoder, self).__init__()
    self.fc1 = nn.Linear(obs_dim, 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, output_dim)
  
  def forward(self, x):
    x = torch.sigmoid(self.fc1(x.float()))
    x = torch.sigmoid(self.fc2(x))
    x = self.fc3(x)
    return x

class PredictNet_1(nn.Module):
  def __init__(self, action_dim, obs_feature_dim, output_dim):
    super(PredictNet_1, self).__init__()
    self.fc1 = nn.Linear(action_dim+obs_feature_dim, 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, output_dim)
  
  def forward(self, action, obs_feature):
    x = torch.cat([action, obs_feature], dim=0)
    x = torch.sigmoid(self.fc1(x.float()))
    x = torch.sigmoid(self.fc2(x))
    x = self.fc3(x)
    return x

class PredictNet_2(nn.Module):
  def __init__(self, obs_feature_dim, action_dim):
    super(PredictNet_2, self).__init__()
    self.fc1 = nn.Linear(2*obs_feature_dim, 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, action_dim)
  
  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=0)
    x = torch.sigmoid(self.fc1(x.float()))
    x = torch.sigmoid(self.fc2(x))
    action_score = self.fc3(x)
    return F.softmax(action_score, dim=-1)


class ICM():
    def __init__(self, obs_dim=2, action_dim=4, obs_feature_dim=4, output_dim=4):
      super(ICM, self).__init__()
      self.obs_dim = obs_dim
      self.action_dim = action_dim
      self.obs_feature_dim = obs_feature_dim
      self.output_dim = output_dim

      self.feature_encoder = FeatureEncoder(obs_dim, obs_feature_dim)
      self.predict_net_1 = PredictNet_1(action_dim, obs_feature_dim, output_dim)
      self.predict_net_2 = PredictNet_2(obs_feature_dim, action_dim)

      self.optimizer1 = torch.optim.Adam(self.predict_net_1.parameters(), lr=1e-3)
      self.optimizer2 = torch.optim.Adam(self.predict_net_2.parameters(), lr=1e-3)

      self.crition = torch.nn.CrossEntropyLoss()

    def get_intrinsic_reward(self, action, state, next_state):
      action = torch.from_numpy(action)
      state_feature = self.feature_encoder(torch.from_numpy(state))
      
      next_state_feature = self.feature_encoder(torch.from_numpy(next_state))

      next_state_feature_pred = self.predict_net_1.forward(action, state_feature)
      intrinsic_reward = torch.norm(next_state_feature - next_state_feature_pred).item()

      action_pred = self.predict_net_2.forward(state_feature, next_state_feature)
      importance = torch.exp( -torch.norm(action - action_pred) ).item()

      # Train network 1
      #print("next state ", next_state_feature)
      #print("next state pred ", next_state_feature_pred)
      loss1 = torch.norm(next_state_feature - next_state_feature_pred)
      self.optimizer1.zero_grad()
      loss1.backward()
      self.optimizer1.step()

      # Train network 2
      """print("action ", action)
      print("action pred ", action_pred)
      loss2 = self.crition(action_pred, torch.tensor([0, 1, 0, 0])) #torch.norm(action - action_pred)
      
      loss2.backward()
      self.optimizer2.step()
      self.optimizer2.zero_grad()"""

      #intrinsic_reward = intrinsic_reward*importance
      return intrinsic_reward

