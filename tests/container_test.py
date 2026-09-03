from app.config.container import Container

container = Container()

manager = container.memory_manager()
registry = container.tool_registry()

print(manager)
print(registry.keys())