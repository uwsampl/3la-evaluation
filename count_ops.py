import sys
from models.end_to_end_speech_to_text import build_relay_module
from models.end_to_end_speech_to_text import load_weights
def get_args():
    lstm_i2h_w, lstm_h2h_w, lstm_bias, linear_w, linear_bias = load_weights()
    i2h_shape = np.shape(lstm_i2h_w)
    print("i2h_shape: {}".format(i2h_shape))
    assert len(i2h_shape) == 2
    # LSTM packs in four separate weights into one tensor
    print(i2h_shape)
    hidden_size = i2h_shape[0] // 4
    input_size = i2h_shape[1]
    linear_shape = np.shape(linear_bias)
    assert len(linear_shape) == 1
    dense_dim = linear_shape[0]
    print("input size: " + input_size + "\n")
    print("hidden size:" + hidden_size + "\n")
get_args()
# mod = build_relay_module(stuff)

# count_all_ops(mod)



# efficientnet


# resmlp



