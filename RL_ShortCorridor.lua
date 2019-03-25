dofile("Bitmap.lua")
dofile("Graphics.lua")

local EPSILON                                     = 0.1
local GAMMA                                       = 1.0
local EPISODES                                    = 1000

local STATE_START                                 = 1
local STATE_REVERSED                              = 2
local STATE_TERMINAL                              = 4
local REWARD                                      = -1.0
local ACTION_LEFT                                 = 1
local ACTION_RIGHT                                = 2
local ACTIONS                                     = {ACTION_LEFT, ACTION_RIGHT}
local FEATURES                                    = {[ACTION_LEFT] = {0, 1}, [ACTION_RIGHT]= {1, 0}}

local ALPHAS                                      = {math.pow(2.0, -12.0), math.pow(2.0, -13.0), math.pow(2.0, -14.0)}
local COLORS                                      = {RGB_CYAN, RGB_RED, RGB_GREEN}
local NAMES                                       = {"Alpha=2^-12", "Alpha=2^-13", "Alpha=2^-14"}
local RUNS                                        = 100

local ALPHA_BASELINE_OFF                          = math.pow(2.0, -13.0)
local ALPHA_BASELINE_THETA                        = math.pow(2.0, -9.0)
local ALPHA_BASELINE_STATE_VALUE                  = math.pow(2.0, -6.0)
local ALPHA_BASELINE_OFF_NAME                     = "Alpha=2^-13"
local ALPHA_BASELINE_THETA_NAME                   = "Alpha Theta=2^-9"
local ALPHA_BASELINE_STATE_VALUE_NAME             = "Alpha State-Value=2^-6"

local ALPHA_ACTOR_CRITIC_ACTOR                    = math.pow(2.0, -9.0)
local ALPHA_ACTOR_CRITIC_CRITIC                   = math.pow(2.0, -6.0)
local ALPHA_ACTOR_CRITIC_ACTOR_NAME               = "Alpha Actor=2^-9"
local ALPHA_ACTOR_CRITIC_CRITIC_NAME              = "Alpha Critic=2^-6"
local LAMBDA_ACTOR_CRITIC_ACTOR                   = 0.95
local LAMBDA_ACTOR_CRITIC_CRITIC                  = 0.98
local LAMBDA_ACTOR_CRITIC_ACTOR_NAME              = "Lambda Actor=0.95"
local LAMBDA_ACTOR_CRITIC_CRITIC_NAME             = "Lambda Critic=0.98"

local IMAGE_WIDTH                                 = 1000
local IMAGE_HEIGHT                                = 1000
local IMAGE_FILENAME_EXAMPLE13_1                  = "ShortCorridor/ShortCorridor_Example13_1.bmp"
local IMAGE_FILENAME_FIGURE13_1                   = "ShortCorridor/ShortCorridor_Figure13_1.bmp"
local IMAGE_FILENAME_FIGURE13_2                   = "ShortCorridor/ShortCorridor_Figure13_2.bmp"
local IMAGE_FILENAME_FIGURE_ACTOR_CRITIC          = "ShortCorridor/ShortCorridor_Figure_ActorCritic.bmp"

local function TrueValue(epsilon_half)
  return 2.0 * (2.0 - epsilon_half) / (epsilon_half * (epsilon_half - 1))
end

local function Example13_1()
  local func_points = {color = RGB_GREEN}
  local graphs = {funcs = {["Policies"] = func_points}, name_x = "Probability of Right action", name_y = "Value of the 1st state"}
  local points = 10000
  local prob_right_min, prob_right_max = 0.01, 1.0 - 0.01
  local min_y = -100
  local max, max_p
  for i = 1, points do
    local prob_right = prob_right_min + (prob_right_max - prob_right_min) * (i - 1) / (points - 1)
    local true_value = TrueValue(prob_right)
    if true_value > min_y then
      table.insert(func_points, {x = prob_right, y = true_value > min_y and true_value or min_y})
      if not max or true_value > max then
        max = true_value or max
        max_p = prob_right
      end
    end
  end
  
  local epsilon_greedy = {funcs = {}, name_x = "", name_y = ""}
  graphs.funcs["<skip>1"] = {color = RGB_RED, {x = EPSILON / 2.0, y = TrueValue(EPSILON / 2.0), text = "E-Greedy Left"}, skip_KP = false}
  graphs.funcs["<skip>2"] = {color = RGB_CYAN, {x = 1.0 - EPSILON / 2.0, y = TrueValue(1.0 - EPSILON / 2.0), text = "E-Greedy Right"}, skip_KP = false}
  graphs.funcs["<skip>3"] = {color = RGB_WHITE, {x = max_p, y = max, text = "Optimal Stochastic Policy"}, skip_KP = false}
  local bmp = Bitmap.new(IMAGE_WIDTH, IMAGE_HEIGHT, RGB_BLACK)
  DrawGraphs(bmp, graphs, {skip_KP = true, min_x = 0.0, max_x = 1.0, y_line = max, KP_size = 5, min_y = min_y, max_y = 0, div_y = 5})
  bmp:WriteBMP(IMAGE_FILENAME_EXAMPLE13_1)
  
  return max
end

local function CalculateProb(theta, prob)   -- feature looks the same for all states so does the probability
  local total_exp = 0.0
  for _, a in ipairs(ACTIONS) do
    local feature = FEATURES[a]
    local h = 0.0
    for k = 1, #theta do
      h = h + theta[k] * feature[k]
    end
    local exp = math.exp(h)
    prob[a] = exp
    total_exp = total_exp + exp
  end
  
  -- normalize so the sum is 1
  for a, exp in pairs(prob) do
    prob[a] = exp / total_exp
  end
end

local function ChooseAction(s, prob)
  return (math.random() < prob[ACTION_RIGHT]) and ACTION_RIGHT or ACTION_LEFT
end

local function TakeAction(s, a)
  if s == STATE_START then
    s = (a == ACTION_LEFT) and s or (s + 1)
  elseif s == STATE_REVERSED then
    s = (a == ACTION_LEFT) and (s + 1) or (s - 1)
  else
    s = (a == ACTION_LEFT) and (s - 1) or (s + 1)
  end
  
  return s, REWARD
end

local function CalculatePolicyParameterGradient(action, prob)
  local grad_ln_prob = {}
  local feature = FEATURES[action]
  for k, value in ipairs(feature) do
    grad_ln_prob[k] = value
  end
  for _, a in ipairs(ACTIONS) do
    local feature = FEATURES[a]
    local prob_a = prob[a]
    for k, value in ipairs(grad_ln_prob) do
      grad_ln_prob[k] = grad_ln_prob[k] - prob_a * feature[k]
    end
  end
  
  return grad_ln_prob
end

local function REINFORCE(alpha, gamma, episodes, episode_reward)
  local p = EPSILON / 2.0
  local theta = {math.log(p / (1.0 - p)), 0.0}
  local prob = {}
  for episode = 1, episodes do
    CalculateProb(theta, prob)
    
    -- generate an episode
    local s = STATE_START
    local a = ChooseAction(s, prob)
    local trajectory = {{s = s, a = a, r = 0.0}}
    local total_reward = 0.0
    while s ~= STATE_TERMINAL do
      local next_s, r = TakeAction(s, a)
      table.insert(trajectory, {s = next_s, r = r})
      if next_s ~= STATE_TERMINAL then
        a = ChooseAction(next_s, prob)
        trajectory[#trajectory].a = a
      end
      s = next_s
      total_reward = total_reward + r
    end
    episode_reward[episode] = (episode_reward[episode] or 0.0) + total_reward
    
    -- train on that episode    
    local T = #trajectory
    -- calculate return
    local G = {[T] = trajectory[T].r}
    for t = T - 1, 1, -1 do
      G[t] = trajectory[t].r + gamma * G[t + 1]
    end
    local gamma_pow_t = 1.0
    for t = 1, T - 1 do
      local entry = trajectory[t]
      local grad_ln_prob = CalculatePolicyParameterGradient(entry.a, prob)
      -- update policy parameter
      local delta = alpha * gamma_pow_t * G[t]
      gamma_pow_t = gamma_pow_t * gamma
      for k, value in ipairs(theta) do
        theta[k] = value + delta * grad_ln_prob[k]
      end
    end
  end
end

local function REINFORCE_Baseline(alpha_theta, alpha_state_value, gamma, episodes, episode_reward)
  local p = EPSILON / 2.0
  local theta = {math.log(p / (1.0 - p)), 0.0}
  local prob = {}
  local w = 0.0
  for episode = 1, episodes do
    CalculateProb(theta, prob)
    
    -- generate an episode
    local s = STATE_START
    local a = ChooseAction(s, prob)
    local trajectory = {{s = s, a = a, r = 0.0}}
    local total_reward = 0.0
    while s ~= STATE_TERMINAL do
      local next_s, r = TakeAction(s, a)
      table.insert(trajectory, {s = next_s, r = r})
      if next_s ~= STATE_TERMINAL then
        a = ChooseAction(next_s, prob)
        trajectory[#trajectory].a = a
      end
      s = next_s
      total_reward = total_reward + r
    end
    episode_reward[episode] = (episode_reward[episode] or 0.0) + total_reward
    
    -- train on that episode    
    local T = #trajectory
    -- calculate return
    local G = {[T] = trajectory[T].r}
    for t = T - 1, 1, -1 do
      G[t] = trajectory[t].r + gamma * G[t + 1]
    end
    local gamma_pow_t = 1.0
    for t = 1, T - 1 do
      local entry = trajectory[t]
      -- update state-value parameter
      w = w + alpha_state_value * (G[t] - w)
      -- update policy parameter - using already updated w(different than the book!!!)
      -- NOTE: in the book (G[t] - w) before updating w is used(same delta as w update) but the graph looks different
      local grad_ln_prob = CalculatePolicyParameterGradient(entry.a, prob)
      local delta = alpha_theta * gamma_pow_t * (G[t] - w)
      gamma_pow_t = gamma_pow_t * gamma
      for k, value in ipairs(theta) do
        theta[k] = value + delta * grad_ln_prob[k]
      end
    end
  end
end

local function ActorCriticTD0(alpha_actor, alpha_critic, gamma, episodes, episode_reward)
  local p = EPSILON / 2.0
  local theta = {math.log(p / (1.0 - p)), 0.0}
  local prob = {}
  local w = {}
  for s = STATE_START, STATE_TERMINAL do
    w[s] = 0.0
  end
  for episode = 1, episodes do
    CalculateProb(theta, prob)
    
    -- generate an episode
    local s = STATE_START
    local a = ChooseAction(s, prob)
    local trajectory = {{s = s, a = a, r = 0.0}}
    local total_reward = 0.0
    local gamma_pow_t = 1.0
    while s ~= STATE_TERMINAL do
      local next_s, r = TakeAction(s, a)
      table.insert(trajectory, {s = next_s, r = r})
      if next_s ~= STATE_TERMINAL then
        a = ChooseAction(next_s, prob)
        trajectory[#trajectory].a = a
      end
      local delta = r + gamma * w[next_s] - w[s]
      w[s] = w[s] + gamma * alpha_critic * delta
      local grad_ln_prob = CalculatePolicyParameterGradient(a, prob)
      local delta = alpha_actor * gamma_pow_t * delta
      for k, value in ipairs(theta) do
        theta[k] = value + delta * grad_ln_prob[k]
      end
      s = next_s
      total_reward = total_reward + r
      gamma_pow_t = gamma_pow_t * gamma
    end
    episode_reward[episode] = (episode_reward[episode] or 0.0) + total_reward
  end
end

local function ActorCriticEligibilityTraces(alpha_actor, alpha_critic, lambda_actor, lambda_critic, gamma, episodes, episode_reward)
  local p = EPSILON / 2.0
  local theta = {math.log(p / (1.0 - p)), 0.0}
  local prob = {}
  local w = {}
  for s = STATE_START, STATE_TERMINAL do
    w[s] = 0.0
  end
  local z_actor, z_critic = {}, {}
  local decay_actor, decay_critic = lambda_actor * gamma, lambda_critic * gamma
  
  for episode = 1, episodes do
    CalculateProb(theta, prob)
    for s = STATE_START, STATE_TERMINAL do
      z_critic[s] = 0.0
    end
    for s = 1, #theta do
      z_actor[s] = 0.0
    end
    
    -- generate an episode
    local s = STATE_START
    local a = ChooseAction(s, prob)
    local trajectory = {{s = s, a = a, r = 0.0}}
    local total_reward = 0.0
    local gamma_pow = 1.0
    while s ~= STATE_TERMINAL do
      local next_s, r = TakeAction(s, a)
      table.insert(trajectory, {s = next_s, r = r})
      if next_s ~= STATE_TERMINAL then
        a = ChooseAction(next_s, prob)
        trajectory[#trajectory].a = a
      end
      local delta = r + gamma * w[next_s] - w[s]
      local grad_ln_prob = CalculatePolicyParameterGradient(a, prob)
      for k = 1, #z_critic do
        z_critic[k] = decay_critic * z_critic[k]
      end
      z_critic[s] = z_critic[s] + 1.0     -- accumulating traces
      for k = 1, #z_actor do
        z_actor[k] = decay_actor * z_actor[k] + gamma_pow * grad_ln_prob[k]
      end
      for k = 1, #w do
        w[k] = w[k] + alpha_critic * delta * z_critic[k]
      end
      for k = 1, #theta do
        theta[k] = theta[k] + alpha_actor * delta * z_actor[k]
      end
      s = next_s
      total_reward = total_reward + r
      gamma_pow = gamma_pow * gamma
    end
    episode_reward[episode] = (episode_reward[episode] or 0.0) + total_reward
  end
end

local function RegisterFunc(graphs, name, color, episode_reward, runs)
  local func_points = {color = color, sort_idx = name}
  for k, value in ipairs(episode_reward) do
    func_points[k] = {x = k, y = value / runs}
  end
  graphs.funcs[name] = func_points
end

local function Figure13_1(max)
  local graphs = {funcs = {}, name_x = "Episode", name_y = string.format("G0: Total reward on episode(averaged over %d runs)", RUNS)}
  
  for idx, alpha in ipairs(ALPHAS) do
    local name = NAMES[idx]
    local episode_reward = {}
    math.randomseed(0)
    for run = 1, RUNS do
      if run == 1 or RUNS < 10 or run % (RUNS // 10) == 0 then
        print(string.format("Figure 13.1: #%d/%d for %s", run, RUNS, name))
      end
      REINFORCE(alpha, GAMMA, EPISODES, episode_reward)
    end
    RegisterFunc(graphs, name, COLORS[idx], episode_reward, RUNS)
  end
  
  local bmp = Bitmap.new(IMAGE_WIDTH, IMAGE_HEIGHT, RGB_BLACK)
  DrawGraphs(bmp, graphs, {skip_KP = true, y_line = max, div_y = 5, int_x = true})
  bmp:WriteBMP(IMAGE_FILENAME_FIGURE13_1)
end

local function Figure13_2(max)
  local graphs = {funcs = {}, name_x = "Episode", name_y = string.format("G0: Total reward on episode(averaged over %d runs)", RUNS)}
  
  local episode_reward = {}
  math.randomseed(0)
  for run = 1, RUNS do
    if run == 1 or RUNS < 10 or run % (RUNS // 10) == 0 then
      print(string.format("Figure 13.2: #%d/%d for %s", run, RUNS, ALPHA_BASELINE_OFF_NAME))
    end
    REINFORCE(ALPHA_BASELINE_OFF, GAMMA, EPISODES, episode_reward)
  end
  RegisterFunc(graphs, string.format("REINFORCE %s", ALPHA_BASELINE_OFF_NAME), RGB_RED, episode_reward, RUNS)
    
  local episode_reward = {}
  math.randomseed(0)
  for run = 1, RUNS do
    if run == 1 or RUNS < 10 or run % (RUNS // 10) == 0 then
      print(string.format("Figure 13.2: #%d/%d for %s, %s", run, RUNS, ALPHA_BASELINE_THETA_NAME, ALPHA_BASELINE_STATE_VALUE_NAME))
    end
    REINFORCE_Baseline(ALPHA_BASELINE_THETA, ALPHA_BASELINE_STATE_VALUE, GAMMA, EPISODES, episode_reward)
  end
  RegisterFunc(graphs, string.format("REINFORCE with Baseline %s %s", ALPHA_BASELINE_THETA_NAME, ALPHA_BASELINE_STATE_VALUE_NAME), RGB_GREEN, episode_reward, RUNS)
  
  local bmp = Bitmap.new(IMAGE_WIDTH, IMAGE_HEIGHT, RGB_BLACK)
  DrawGraphs(bmp, graphs, {skip_KP = true, y_line = max, div_y = 5, int_x = true})
  bmp:WriteBMP(IMAGE_FILENAME_FIGURE13_2)
  
  return graphs
end

local function ClampAndGetMinMaxY(...)
  local min_y, max_y
  for _, funcs in ipairs({...}) do
    for _, func_pts in pairs(funcs) do
      for _, pt in ipairs(func_pts) do
        min_y = (not min_y or pt.y < min_y) and pt.y or min_y
        max_y = (not max_y or pt.y > max_y) and pt.y or max_y
      end
    end
  end
  
  return min_y, max_y
end

local function FigureActorCritic(max, graphs_right)
  local graphs = {funcs = {}, name_x = "Episode", name_y = string.format("Actor-Critic G0: Total reward on episode(averaged over %d runs) for %s %s", RUNS, ALPHA_ACTOR_CRITIC_ACTOR_NAME, ALPHA_ACTOR_CRITIC_CRITIC_NAME)}
  
  local episode_reward = {}
  math.randomseed(0)
  for run = 1, RUNS do
    if run == 1 or RUNS < 10 or run % (RUNS // 10) == 0 then
      print(string.format("Figure Actor-Critic TD0: #%d/%d for %s %s", run, RUNS, ALPHA_ACTOR_CRITIC_ACTOR_NAME, ALPHA_ACTOR_CRITIC_CRITIC_NAME))
    end
    ActorCriticTD0(ALPHA_ACTOR_CRITIC_ACTOR, ALPHA_ACTOR_CRITIC_CRITIC, GAMMA, EPISODES, episode_reward)
  end
  RegisterFunc(graphs, string.format("Actor Critic TD0"), RGB_RED, episode_reward, RUNS)
    
  local episode_reward = {}
  math.randomseed(0)
  for run = 1, RUNS do
    if run == 1 or RUNS < 10 or run % (RUNS // 10) == 0 then
      print(string.format("Figure Actor-Critic Eligibility Traces: #%d/%d for %s %s %s %s", run, RUNS, ALPHA_ACTOR_CRITIC_ACTOR_NAME, ALPHA_ACTOR_CRITIC_CRITIC_NAME, LAMBDA_ACTOR_CRITIC_ACTOR_NAME, LAMBDA_ACTOR_CRITIC_CRITIC_NAME))
    end
    ActorCriticEligibilityTraces(ALPHA_ACTOR_CRITIC_ACTOR, ALPHA_ACTOR_CRITIC_CRITIC, LAMBDA_ACTOR_CRITIC_ACTOR, LAMBDA_ACTOR_CRITIC_CRITIC, GAMMA, EPISODES, episode_reward)
  end
  RegisterFunc(graphs, string.format("Eligibility Traces %s %s", LAMBDA_ACTOR_CRITIC_ACTOR_NAME, LAMBDA_ACTOR_CRITIC_CRITIC_NAME), RGB_GREEN, episode_reward, RUNS)
  
  local min_y, max_y = ClampAndGetMinMaxY(graphs.funcs, graphs_right.funcs)
  
  local bmp = Bitmap.new(2 * IMAGE_WIDTH, IMAGE_HEIGHT, RGB_BLACK)
  DrawGraphs(bmp, graphs, {width = IMAGE_WIDTH, height = IMAGE_HEIGHT, skip_KP = true, min_y = min_y, max_y = max_y, y_line = max, int_x = true})
  DrawGraphs(bmp, graphs_right, {width = IMAGE_WIDTH, height = IMAGE_HEIGHT, skip_KP = true, start_x = IMAGE_WIDTH, start_y = 0, min_y = min_y, max_y = max_y, y_line = max, int_x = true})
  bmp:WriteBMP(IMAGE_FILENAME_FIGURE_ACTOR_CRITIC)
end

local max = Example13_1()
--local max = -11.6
Figure13_1(max)
local graphs_Fig13_2 = Figure13_2(max)
FigureActorCritic(max, graphs_Fig13_2)