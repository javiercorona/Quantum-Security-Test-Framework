# Quantum Security Test Framework

![Quantum Security Banner](https://via.placeholder.com/800x200?text=Quantum+Security+Test+Framework)

A cutting-edge security testing tool with quantum-resistant cryptography, federated testing capabilities, and real-time monitoring.

## Features

- **Quantum-Safe Cryptography**: Integration with post-quantum algorithms (Kyber, Dilithium)
- **Multi-Test Support**: XSS, Network Scanning, Header Analysis, API Testing
- **Federated Testing**: Distribute tests across multiple nodes
- **Real-Time Web UI**: Live monitoring with WebSocket updates
- **Comprehensive Reporting**: PDF, HTML, and JSON output formats
- **Plugin System**: Extensible architecture for custom tests
- **CI/CD Integration**: Built for automated security pipelines

## Installation

### Prerequisites

- Python 3.8+
- Chrome/Chromium browser
- Redis (for federated mode)
- Nmap (for advanced port scanning)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-security-framework.git
cd quantum-security-framework

# Install dependencies
pip install -r requirements.txt

# Run a sample test
python -m framework.run_example
```

## Usage

### Basic Test

```python
from framework.core import TestConfig, TestType, SecurityLevel
from framework.runner import QuantumTestRunner
import asyncio

config = TestConfig(
    test_type=TestType.XSS,
    target="http://example.com/login",
    security_level=SecurityLevel.QUANTUM,
    payloads=["<script>alert('XSS')</script>", "<img src=x onerror=alert(1)>"]
)

async def run_test():
    runner = QuantumTestRunner(config)
    try:
        results = await runner.run_test()
        print(json.dumps(results, indent=2))
    finally:
        runner.cleanup()

asyncio.run(run_test())
```

### Federated Testing

```python
config = TestConfig(
    test_type=TestType.NETWORK,
    target="http://example.com",
    federated_nodes=["node1.example.com:5555", "node2.example.com:5555"]
)
```

### Web UI

Start the dashboard with:

```bash
python -m framework.web.ui
```

Then access: http://localhost:8080

## Configuration

The framework can be configured via `config.yaml`:

```yaml
defaults:
  timeout: 15
  headless: true
  output_dir: ./reports
  security_level: QUANTUM

plugins:
  enabled:
    - xss_detector
    - sql_injection
    - tls_analyzer

federated:
  nodes:
    - node1.example.com:5555
    - node2.example.com:5555
  redis_url: redis://localhost:6379
```

## Plugin Development

Create custom tests by extending the base plugin class:

1. Create a new Python file in `plugins/`
2. Decorate your class with `@PluginRegistry.register`

Example plugin:

```python
from framework.plugins import PluginRegistry

@PluginRegistry.register("example_plugin", "Example test plugin")
class ExamplePlugin:
    def __init__(self, config):
        self.config = config
    
    async def run(self):
        return {
            "status": "completed",
            "result": "Plugin test completed successfully",
            "timestamp": datetime.now().isoformat()
        }
```

## Architecture

```
quantum-security-framework/
âââ framework/
â   âââ core/               # Core classes and types
â   âââ crypto/             # Quantum cryptography
â   âââ plugins/            # Plugin system
â   âââ scanner/            # Network scanning
â   âââ web/                # Web UI components
â   âââ runner.py           # Main test runner
âââ plugins/                # Custom plugins
âââ config.yaml             # Framework configuration
âââ requirements.txt        # Python dependencies
âââ run_example.py          # Example test scenarios
```

## Security Considerations

- Always test against staging environments first
- The framework performs actual attacks (XSS, SQLi, etc.)
- Use proper authorization for all tests
- Quantum crypto keys are rotated hourly by default

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Apache License 
