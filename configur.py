data_path = 'data/'

# data parameters
start_date = "2021-07-01"
end_date = "2023-06-25"
interval = "60m"
data_mode = "load" # download
train_test_size = 0.8

# series parameters
target = "Close"
ma_short_period = 7
ma_long_period = 14
oscillator_period = 14

# model parameters
batch_size = 64
learning_rate = 0.001
epochs = 100