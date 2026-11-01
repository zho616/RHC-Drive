"""Compact privileged vehicle condition, read from the spawned physical vehicle."""
import numpy as np

# Name, fixed normalization scale, physical unit. No vehicle-type identifier.
ORACLE_SCHEMA = (
    ('mass', 1000., 'kg'),
    ('max_steering', 60., 'degree'),
    ('wheel_friction', 1., 'Bullet friction slip'),
    ('length', 5., 'm'),
    ('width', 2., 'm'),
    ('wheelbase', 3., 'm'),
    ('track', 2., 'm'),
    ('wheel_radius', .5, 'm'),
    ('max_engine_force', 1000., 'N per wheel'),
    ('max_brake_force', 100., 'Bullet brake setting per wheel'),
)
ORACLE_KEYS = tuple('log_chassis_' + name for name, _, _ in ORACLE_SCHEMA)
ORACLE_SCALE = tuple(scale for _, scale, _ in ORACLE_SCHEMA)


def read_vehicle_parameters(vehicle):
    """Read actual wheel attachment geometry and control settings, once per reset."""
    wheels = [vehicle.system.getWheel(i) for i in range(vehicle.system.getNumWheels())]
    if len(wheels) != 4:
        raise ValueError('Compact oracle expects a four-wheel vehicle')
    positions = np.array([tuple(w.getChassisConnectionPointCs()) for w in wheels], dtype=float)
    front_track = abs(positions[0, 0] - positions[1, 0])
    rear_track = abs(positions[2, 0] - positions[3, 0])
    radius = np.array([w.getWheelRadius() for w in wheels])
    friction = np.array([w.getFrictionSlip() for w in wheels])
    if not (np.isclose(front_track, rear_track) and np.allclose(radius, radius[0])
            and np.allclose(friction, friction[0])):
        raise ValueError('Compact oracle cannot collapse unequal tracks/radii/friction into one value')
    values = (
        vehicle.body.getMass(), vehicle.max_steering, friction[0],
        vehicle.LENGTH, vehicle.WIDTH,
        abs(positions[:2, 1].mean() - positions[2:, 1].mean()),
        front_track, radius[0],
        vehicle.config['max_engine_force'], vehicle.config['max_brake_force'],
    )
    if not np.isfinite(values).all() or (np.asarray(values) <= 0).any():
        raise ValueError('Vehicle oracle contains invalid physical parameters')
    return {key: np.float32(value) for key, value in zip(ORACLE_KEYS, values)}
