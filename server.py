# import flwr as fl
#
# fl.server.start_server(server_address="127.0.0.1:8080")

import flwr as fl
import sys
import numpy as np

fl.common.logger.configure(identifier="myFlowerExperiment", filename="log_server.txt")


class SaveModelStrategy(fl.server.strategy.FedMedian):

    def aggregate_fit(
            self,
            rnd,
            results,
            failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            # np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights
    # fed=fl.server.strategy.FedAvg(min_fit_clients=3,


# min_available_clients=3)
strategy = SaveModelStrategy(min_fit_clients=3, min_available_clients=3, min_evaluate_clients=3)
# Create strategy and run server
# strategy = fl.server.strategy.FedAvg(

#     min_available_clients=3,  # Minimum number of clients that need to be connected to the server before a training round can start
# )

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address="127.0.0.1:8080",

    config=fl.server.ServerConfig(num_rounds=2),
    # grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy
)
