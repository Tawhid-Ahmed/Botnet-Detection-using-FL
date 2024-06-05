import json

import pika

# Connect to the RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare a queue to receive messages from
channel.queue_declare(queue='my_queue')

file_name = 'response.csv'


def callback(ch, method, properties, data):
    # Get the data from the request
    data = json.loads(data)

    # Extract the fields: client_id, accuracy, and loss
    client_id = data['client_id']
    accuracy = data['accuracy']
    loss = data['loss']
    c_round = data['round']
    cr = data['cr']

    # Parse the classification report
    lines = cr.strip().split('\n')
    data_cr = [line.split() for line in lines[2:-4]]  # Extract rows with class-wise metrics

    # Create a dictionary with class names as keys and metrics as values
    class_metrics = {row[0]: list(map(float, row[1:])) for row in data_cr}

    # Convert the dictionary to a list of key-value pairs
    key_value_pairs = [(f"{class_name}_{metric_name}", metric_value) for class_name, metrics in class_metrics.items()
                       for metric_name, metric_value in zip(['precision', 'recall', 'f1-score', 'support'], metrics)]

    values = ''
    columns = ''

    for key, value in key_value_pairs:
        values += f',{value}'
        columns += f',{key}'

    print(f'{client_id},{c_round}')

    # Process the data (you can perform any desired operations here)
    # For demonstration, let's save the data to a file
    with open(file_name, 'a') as file:
        file.write(f'{client_id},{accuracy},{loss},{c_round}{values}\n')

    # Return a response
    print(f"Received: Client {client_id} Round {c_round}")


if __name__ == '__main__':
    with open(file_name, 'w') as file:
        file.write(
            f'Client_ID,Accuracy,Loss,Round,benign_precision,benign_recall,benign_f1-score,benign_support,mirai_udp_precision,mirai_udp_recall,mirai_udp_f1-score,mirai_udp_support,gafgyt_combo_precision,gafgyt_combo_recall,gafgyt_combo_f1-score,gafgyt_combo_support,gafgyt_junk_precision,gafgyt_junk_recall,gafgyt_junk_f1-score,gafgyt_junk_support,gafgyt_scan_precision,gafgyt_scan_recall,gafgyt_scan_f1-score,gafgyt_scan_support,gafgyt_tcp_precision,gafgyt_tcp_recall,gafgyt_tcp_f1-score,gafgyt_tcp_support,gafgyt_udp_precision,gafgyt_udp_recall,gafgyt_udp_f1-score,gafgyt_udp_support,mirai_ack_precision,mirai_ack_recall,mirai_ack_f1-score,mirai_ack_support,mirai_scan_precision,mirai_scan_recall,mirai_scan_f1-score,mirai_scan_support,mirai_syn_precision,mirai_syn_recall,mirai_syn_f1-score,mirai_syn_support,mirai_udpplain_precision,mirai_udpplain_recall,mirai_udpplain_f1-score,mirai_udpplain_support\n')
    # Set up a consumer that calls the callback function
    channel.basic_consume(queue='my_queue', on_message_callback=callback, auto_ack=True)

    # Start listening for messages
    print('Waiting for messages. To exit, press CTRL+C')
    channel.start_consuming()
