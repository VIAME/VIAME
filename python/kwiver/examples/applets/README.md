# Python KWIVER Applets

This directory contains examples and documentation for creating KWIVER applets in Python.

## What are KWIVER Applets?

KWIVER applets are command-line tools that integrate with the KWIVER framework. They provide a standardized way to create tools with:
- Consistent command-line interfaces
- Plugin-based discovery and loading
- Configuration file support
- Integration with the KWIVER ecosystem

## Creating a Python Applet

### Basic Structure

A Python applet must inherit from `KwiverApplet` and implement the `run()` method:

```python
from kwiver.vital.applets import KwiverApplet, register_applet

class MyApplet(KwiverApplet):
    def run(self):
        """Main entry point for the applet."""
        print(f"Hello from {self.applet_name()}!")
        return 0  # Return 0 for success

def register_plugins():
    """Register the applet with KWIVER."""
    register_applet(
        MyApplet,
        plugin_name="my-applet",
        description="My custom applet"
    )
```

### Key Methods

#### Required Methods

- **`run()`** - Main entry point that executes the applet logic. Returns an integer status code (0 = success).

#### Optional Methods

- **`add_command_options()`** - Define command-line options for your applet
- **`set_configuration(config)`** - Set applet configuration from a ConfigBlock
- **`get_configuration()`** - Return current configuration as a ConfigBlock
- **`initialize()`** - Initialize internal state

#### Helper Methods Available

- **`applet_name()`** - Get the name of the applet as invoked
- **`applet_args()`** - Get the original command-line arguments
- **`wrap_text(text)`** - Wrap text into a formatted block
- **`find_configuration(filename)`** - Find and read a config file from KWIVER config paths

## Registration

To make your applet discoverable by KWIVER, you need to register it using the plugin system:

```python
from kwiver.vital.applets import register_applet

def register_plugins():
    register_applet(
        MyAppletClass,
        plugin_name="my-applet-name",
        description="Description of what my applet does"
    )
```

The `register_plugins()` function should be specified as an entry point in your `setup.py` or `pyproject.toml`:

### Using setup.py

```python
setup(
    name="my-kwiver-applet",
    entry_points={
        'kwiver.python_plugin_registration': [
            'my_applet = my_package.my_applet:register_plugins'
        ]
    }
)
```

### Using pyproject.toml

```toml
[project.entry-points."kwiver.python_plugin_registration"]
my_applet = "my_package.my_applet:register_plugins"
```

## Running Your Applet

Once registered, your applet can be run using the `kwiver` command:

```bash
kwiver my-applet-name [options]
```

You can also run it directly for testing:

```bash
python my_applet.py [options]
```

## Configuration

Applets can be configured in two ways:

### 1. Configuration Files

Create a configuration file (e.g., `my-applet.conf`):

```
# My applet configuration
input = /path/to/input
output = /path/to/output
verbose = true
```

Load it in your applet:

```python
def run(self):
    config = self.find_configuration("my-applet.conf")
    self.set_configuration(config)
    # ... rest of implementation
```

### 2. Command-Line Options

Parse command-line arguments in your `run()` method:

```python
def run(self):
    args = self.applet_args()

    for i, arg in enumerate(args):
        if arg == "--input" and i + 1 < len(args):
            self.input_file = args[i + 1]
        # ... more argument parsing
```

## Example

See `example_python_applet.py` for a complete, working example that demonstrates:
- Basic applet structure
- Command-line argument parsing
- Configuration management
- Help text generation
- Return code handling

Run the example:

```bash
# Show help
python example_python_applet.py --help

# Run with options
python example_python_applet.py -v -i input.txt -o output.txt
```

## Best Practices

1. **Return Codes**: Always return 0 for success, non-zero for errors
2. **Error Handling**: Wrap your main logic in try/except and return appropriate error codes
3. **Configuration**: Use ConfigBlocks for complex configuration
4. **Help Text**: Provide clear help text using `wrap_text()` for proper formatting
5. **Logging**: Use KWIVER's logging system for consistent output
6. **Testing**: Test your applet both standalone and through the `kwiver` command

## Advanced Topics

### Accessing KWIVER Algorithms

Your Python applet can use any KWIVER algorithm:

```python
from kwiver.vital.algo import ImageIO

def run(self):
    image_io = ImageIO.create("vxl")
    image = image_io.load("input.png")
    # Process image...
    return 0
```

### Using with Pipelines

Applets can launch or interact with KWIVER pipelines:

```python
from kwiver.sprokit.pipeline import Pipeline

def run(self):
    # Load and run a pipeline
    pipe = Pipeline()
    pipe.load_from_file("my_pipeline.pipe")
    pipe.start()
    pipe.wait()
    return 0
```

## Troubleshooting

### Applet Not Found

If `kwiver my-applet` says the applet is not found:
1. Verify your package is installed with the entry point
2. Check `kwiver plugin-explorer --applets` to see registered applets
3. Ensure KWIVER_PYTHON_PATH includes your package

### Import Errors

If you get import errors:
1. Ensure KWIVER Python bindings are built and installed
2. Check your PYTHONPATH includes the KWIVER Python modules
3. Verify all dependencies are installed

## Further Reading

- KWIVER Documentation: https://kwiver.readthedocs.io/
- KWIVER Applets (C++): `packages/kwiver/vital/applets/README.md`
- Python Processes: See KWIVER's Python process documentation
