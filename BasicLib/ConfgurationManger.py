"""
This module manages a set of parameters. The module is designed to easily import a ini parameter file. All value's
are automatically parsed to their logical type. For example:
    "Hello World" > str
    'Hello World' > str
    458.25 > float
    458 > int
    [1, 2, 8] > list with int's
    [(5, 8), 0.8] > list with a tuple and a float
    Hello World > ERROR. A string without " " of ' ' results in a fault


#########
# USAGE #
#########

# config.ini before
###################

[Section1]
parameter1 = 45
parameter2 = 'Test'

[Section2]
nr1 = 456987
parameter3 = 654



# run code:
###########

# load and display an existing parameter set
config = ConfigurationManager('config.ini')
config.print_parameter_set()

# a new parameter set could be created as following:
#config = ConfigurationManager()
#config.clear_parameter_set()

# get value
value = config['Section1']['parameter1']

# change value if 'parameter2' exists
# add parameter if 'parameter2' didn't exist
# function will give an error if Section1 does not exist
config['Section1']['parameter2'] = 45678.258

# add parameter.
config.add_parameter('Section2', 'parameter4', {'test': 123})

# add a new section
config.add_section('new_section')

# add two parameters to 'new_section'
config['new_section']['parameter1'] = [('Hello', 'World'), 123, 1.023]
config.add_parameter('new_section', 'parameter2', [1, 2, 'abc'])

# display parameter set
config.print_parameter_set()

# store configuration. Note that al command will be erased
config.save('config.ini')



# config.ini AFTER
##################

[Section1]
parameter1 = 45
parameter2 = 45678.258

[Section2]
nr1 = 456987
parameter3 = 654
parameter4 = {'test': 123}

[new_section]
parameter1 = [('Hello', 'World'), 123, 1.023]
parameter2 = [1, 2, 'abc']

"""

import configparser
import ast
import pprint


class ConfigurationManager:
    """
    This class manages a set of parameters. A new parameter set can be created and saved, or an existing one could be
    loaded, changed and saved. Note that comments in the parameter file will disappear after a parameter set is saved.
    All value's loaded form an ini file are automatically parsed to their logical type. For example:
        "Hello World" > str
        'Hello World' > str
        458.25 > float
        458 > int
        [1, 2, 8] > list with int's
        [(5, 8), 0.8] > list with a tuple and a float
        Hello World > ERROR. A string without " " of ' ' results in a fault
    """
    def __init__(self, ini_file_location=""):
        self.parameter_set = {}
        if ini_file_location != "":
            self.load(ini_file_location)

    def __getitem__(self, section):
        """
        Returns a parameter set section as parameter_set['Section_name']
        :param section: section name
        :return: parameter value
        """
        return self.get_section(section)

    def get_section(self, section):
        """
        Returns a parameter set section
        :param section: section name
        :return: parameter value
        """
        value = self.parameter_set.get(section)
        if value is None:
            raise NameError("Section \'" + str(section) + "\' does not exist")
        return value

    def add_parameter(self, section, key, value, duplication_detect=False):
        """
        Add's a parameter to a section.
        :param section: section name
        :param key: parameter name
        :param value: parameter value
        :param duplication_detect: check if parameter already exists when True. Throws an NameError if parameter exists.
        """
        key = key.lower()
        if self.get_section(section).get(key) is not None or duplication_detect:
            raise NameError("Key \'" + str(section, key) + "\' does already exist")
        self.parameter_set[section][key] = value

    def add_section(self, section, duplication_detect=False):
        """
        Add's a new section to the parameter set
        :param section: section name
        :param duplication_detect: check if section already exists when True. Throws an NameError if section exists.
        """
        if self.parameter_set.get(section) is not None and duplication_detect:
            raise NameError("Section \'" + str(section) + "\' does already exist")
        self.parameter_set[section] = {}

    def clear_parameter_set(self):
        """Clear all parameters"""
        self.parameter_set = {}

    def load(self, ini_file_location):
        """
        Load ini file in to cleared parameter set.
        :param ini_file_location: ini file location
        """
        self.clear_parameter_set()

        file_parser = configparser.ConfigParser()
        if len(file_parser.read(ini_file_location)) == 0:
            raise FileNotFoundError("Configuration file '{}' not found".format(ini_file_location))
        sections = file_parser.sections()
        for section in sections:
            self.add_section(section)
            for key in file_parser[section]:
                value = ast.literal_eval(file_parser[section][key])
                self.parameter_set[section][key] = value

    def save(self, ini_file_location):
        """
        Saves parameter set to ini file location. File will be cleared before saving.
        :param ini_file_location: ini file location
        """
        file_parser = configparser.ConfigParser()
        for section in self.parameter_set:
            file_parser[section] = {}
            for key in self.parameter_set[section]:
                value = self.parameter_set[section][key]
                if type(value) is str:
                    value = repr(value)
                file_parser[section][key] = str(value)

        with open(ini_file_location, 'w') as ini_file:
            file_parser.write(ini_file)

    def get_parameter_set_as_pretty_string(self, parameter_value_position_offset=25):
        """
        Returns the active parameter set as a pretty, printable string.
        :param parameter_value_position_offset: sets an offset position for the parameter value's
        """
        string = ""
        value_print_format = "{:" + str(parameter_value_position_offset) + "} = {}\n"
        for section, section_dict in self.parameter_set.items():
            string += "[{}]\n".format(section)
            for parameter, value in section_dict.items():
                string += value_print_format.format(parameter, pprint.pformat(value))
            string += "\n"
        return string

    def print_parameter_set(self):
        """Display parameter set"""
        print(self.get_parameter_set_as_pretty_string())
