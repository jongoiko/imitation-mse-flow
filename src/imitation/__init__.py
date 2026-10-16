import sys
import types

if "mujoco_py" not in sys.modules:
    _fake = types.ModuleType("mujoco_py")

    class MujocoException(Exception):
        pass

    _fake.MujocoException = MujocoException
    _fake.builder = types.SimpleNamespace(MujocoException=MujocoException)
    sys.modules["mujoco_py"] = _fake
