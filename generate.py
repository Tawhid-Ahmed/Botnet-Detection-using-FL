import random

no_of_containers = 3
min_fit_clients = 3
initial_delay = 40
general_delay = 10

file_content = ''

excluded_values = [3, 7]

# Function to generate a random number within the range while excluding certain values
def generate_random_with_exclusion(min_value, max_value, excluded_values):
    while True:
        random_number = random.randint(min_value, max_value)
        if random_number not in excluded_values:
            return random_number


for i in range(no_of_containers):
    low = random.randint(0, 1000)
    top = random.randint(1000, 2000)
    count = random.randint(500, 500)
    client_id = i + 1
    epochs = random.randint(1, 1)
    batch_size = random.randint(1, 1)
    dataset_num = generate_random_with_exclusion(1, 9, excluded_values)
    text_to_write = f'start "C{client_id}" docker run --rm --name=fl-server-container{client_id} -v "D:\\BUET\\2023-04\\Distributed Systems\\Project\\Codes\\ProjectTest\\dataset":/app/dataset -e low={low} -e top={top} -e count={count} -e client_id={client_id} -e dataset_num={dataset_num} -e epochs={epochs} -e batch_size={batch_size} fl-server\n'

    # if i == min_fit_clients:
    #     text_to_write += f'timeout /t {initial_delay}\n'
    # elif i > min_fit_clients and i != no_of_containers - 1:
    #     text_to_write += f'timeout /t {general_delay}\n'

    file_content += text_to_write

# Define the file path
file_path = "run_multiple_generated.bat"

with open(file_path, 'w', encoding='utf-8') as file:
    # Write the text to the file
    file.write(file_content)

# Display a message indicating the file has been written
print(f"Text has been written to {file_path}.")
