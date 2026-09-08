"""B0 ball-only plant prototype; no robot controller, training task or admission."""

from dataclasses import dataclass
import math

from mjlab_microduck.first_attempt_smoke import require

STAGES = (
    ('B0','ball/foot geometry and passive contact dynamics'),
    ('B1','flat-ground stance and disturbance recovery'),
    ('B2','placed-on fixed-ball support'),
    ('B3','placed-on prescribed slow rolling'),
    ('B4','placed-on freely rolling ball'),
    ('B5','free-ball disturbances and bounded direction control'),
    ('B6','mount/dismount and whole-controller skill retention'),
)


@dataclass(frozen=True)
class FootballPlant:
    # Proposed nominal size-5-scale shell, not a calibrated inflated football.
    radius_m: float = .11
    mass_kg: float = .43
    sliding_friction: float = .6
    step_s: float = .002

    def __post_init__(self):
        for value,low,high in ((self.radius_m,.02,.30),(self.mass_kg,.005,2.),
                               (self.sliding_friction,.05,2.),(self.step_s,.0005,.005)):
            require(type(value) in (int,float) and math.isfinite(value) and low <= value <= high,
                    'bounded finite ball plant parameter')

    @property
    def inertia_kg_m2(self):
        return (2/3)*self.mass_kg*self.radius_m**2

    def provenance(self):
        return dict(radius_m=self.radius_m,mass_kg=self.mass_kg,inertia_kg_m2=self.inertia_kg_m2,
            inertia_model='thin spherical shell; not measured',contact_model='rigid sphere with soft solver contacts',
            pressure_or_deformation_calibrated=False,robot_included=False,policy_acceptance=False,
            physical_motion_authorized=False)


def upper_surface(radius_m,x_m,y_m):
    """Geometric surface and slope only; not foot reachability or stable contact."""
    require(all(type(v) in (int,float) and math.isfinite(v) for v in (radius_m,x_m,y_m))
            and radius_m > 0,'finite sphere coordinates')
    radial = math.hypot(x_m,y_m)
    require(radial < radius_m,'point strictly inside upper hemisphere projection')
    return dict(height_above_floor_m=radius_m+math.sqrt(radius_m**2-radial**2),
                normal_tilt_rad=math.asin(radial/radius_m),stance_feasibility_verified=False)


def ball_world_xml(plant=FootballPlant(),*,support='free'):
    """Compile/passive-test only. Fixed support must not masquerade as free rolling."""
    require(type(plant) is FootballPlant,'typed plant'); plant.__post_init__()
    require(support in ('fixed','free'),'explicit support mode')
    joint = '<freejoint name="football_free"/>' if support == 'free' else ''
    r,m,i = plant.radius_m,plant.mass_kg,plant.inertia_kg_m2
    return f'''<mujoco model="football_b0_{support}">
  <option timestep="{plant.step_s}" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1" friction="{plant.sliding_friction} 0 0" condim="3"/>
    <body name="football" pos="0 0 {r}">
      {joint}
      <inertial pos="0 0 0" mass="{m}" diaginertia="{i} {i} {i}"/>
      <geom name="football_surface" type="sphere" size="{r}" friction="{plant.sliding_friction} 0 0" condim="3"/>
    </body>
  </worldbody>
</mujoco>'''
