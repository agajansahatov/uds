# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Read the contents of the requirements.txt file
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

    # Extract the package names and versions
    dependencies = [req.split('==') for req in requirements]

    # Generate the install_requires list with version numbers
    install_requires = [f"{dependency[0]}=={dependency[1]}" for dependency in dependencies]

    # Print the install_requires list
    print(install_requires)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
