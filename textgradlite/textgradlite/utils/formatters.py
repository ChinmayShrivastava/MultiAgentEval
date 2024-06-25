def iterative_formatter(data, indent=0):
    formatted_string = ""
    for key, value in data.items():
        if indent == 0:
            formatted_string += "----------\n"
        if isinstance(value, dict):
            formatted_string += " " * indent + f"{key}:\n"
            formatted_string += iterative_formatter(value, indent + 4)
        elif isinstance(value, list):
            formatted_string += " " * indent + f"{key}:\n"
            for item in value:
                formatted_string += iterative_formatter(item, indent + 4)
        else:
            formatted_string += " " * indent + f"{key}: {value}\n"
    return formatted_string

# if __name__ == "__main__":
#     print(iterative_formatter(eg_state))