import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from spektral.layers import GCNConv, GlobalAvgPool

def create_PPI_model(n_features=9):
    # Inputs for protein 1
    node_input = Input(shape=(None, n_features), name='node_input')
    adj_input = Input(shape=(None, None), dtype=tf.float32, sparse=True, name='adj_input')
    segment_ids = Input(shape=(None,), dtype=tf.int32, name='segment_ids')

    dropout_rate = 0.5
    gc1 = GCNConv(256, activation='relu', name='gcn_conv')([node_input, adj_input])
    gc1 = Dropout(dropout_rate)(gc1)
    gc1 = GCNConv(256, activation='relu', name='gcn_conv_1')([gc1, adj_input])
    gc1 = Dropout(dropout_rate)(gc1)
    gc1 = GCNConv(256, activation='relu', name='gcn_conv_2')([gc1, adj_input])
    gc1 = Dropout(dropout_rate)(gc1)


    pool = GlobalAvgPool()([gc1, segment_ids])

    # Final prediction layer
    x = Dense(1000, activation='relu', name='dense1')(pool)
    output = Dense(1, activation='sigmoid', name='PPI_output')(x)

    # Create the final model
    model = Model(inputs=[node_input, adj_input, segment_ids],
                  outputs=output)

    return model

def main():
    model = create_PPI_model()
    model.summary()

if __name__ == "__main__":
    main()

