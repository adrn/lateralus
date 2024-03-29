"""
Borrowed and adapted from the adrn/gaia-actions project.

This is a bit of a hack to just compute actions in a set of potentials around a fiducial
model for Elise.

"""

# Standard library
import os

os.environ["OMP_NUM_THREADS"] = "1"
import logging
import pathlib
import sys

# Third-party
from astropy.utils import iers

iers.conf.auto_download = False
import astropy.table as at
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import h5py
from pyia import GaiaData

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.dynamics.actionangle.tests.staeckel_helpers import (
    galpy_find_actions_staeckel as find_actions_staeckel
)
from schwimmbad.utils import batch_tasks

logger = logging.getLogger(__name__)
logging.basicConfig()


def worker(task):
    (i, j), idx, galcen, meta, pot, cache_file, group_name, ids = task
    ids = ids[idx]
    galcen = galcen[idx]

    w0 = gd.PhaseSpacePosition(galcen.cartesian)

    logger.debug(f"Worker {i}-{j}: running {j-i} tasks now")

    # Set up data containers:
    all_data = {}
    for k, info in meta.items():
        if k == 'source_id':
            all_data[k] = ids
        else:
            shape = (len(ids),) + info["shape"][1:]
            all_data[k] = np.full(shape, np.nan)

    for n in range(len(galcen)):
        all_data["source_id"][n] = ids[n]
        all_data["xyz"][n] = galcen.data.xyz[:, n].to_value(meta["xyz"]["unit"])
        all_data["vxyz"][n] = galcen.velocity.d_xyz[:, n].to_value(
            meta["vxyz"]["unit"]
        )

        try:
            aaf = find_actions_staeckel(pot, w0[n])[0]
        except Exception as e:
            logger.error(f"Failed to pre-compute actions {i}\n{str(e)}")
            continue

        T = np.abs(2 * np.pi / aaf["freqs"].min()).to(u.Gyr)
        try:
            orbit = pot.integrate_orbit(
                w0[n],
                dt=0.5 * u.Myr,
                t1=0 * u.Myr,
                t2=T,
                Integrator=gi.DOPRI853Integrator,
            )
        except Exception as e:
            logger.error(f"Failed to integrate orbit {i}\n{str(e)}")
            continue

        # Compute actions / frequencies / angles
        try:
            res = find_actions_staeckel(pot, orbit)[0]

            all_data["actions"][n] = res["actions"].to_value(
                meta["actions"]["unit"]
            )
            all_data["angles"][n] = res["angles"].to_value(
                meta["angles"]["unit"]
            )
            all_data["freqs"][n] = res["freqs"].to_value(meta["freqs"]["unit"], u.dimensionless_angles())
        except Exception as e:
            logger.error(f"Failed to compute mean actions {i}\n{str(e)}")

        # Other various things:
        # try:
        #     rper = orbit.pericenter(approximate=True).to_value(
        #         meta["r_per"]["unit"]
        #     )
        #     rapo = orbit.apocenter(approximate=True).to_value(
        #         meta["r_apo"]["unit"]
        #     )

        #     all_data["z_max"][n] = orbit.zmax(approximate=True).to_value(
        #         meta["z_max"]["unit"]
        #     )
        #     all_data["r_per"][n] = rper
        #     all_data["r_apo"][n] = rapo
        #     all_data["ecc"][n] = (rapo - rper) / (rapo + rper)
        # except Exception as e:
        #     logger.error(f"Failed to compute zmax peri apo for orbit {i}\n{e}")

        # # Lz and E
        # try:
        #     all_data["L"][n] = np.mean(
        #         orbit.angular_momentum().to_value(meta["L"]["unit"]), axis=1
        #     )
        #     all_data["E"][n] = np.mean(
        #         orbit.energy().to_value(meta["E"]["unit"])
        #     )
        # except Exception as e:
        #     logger.error(f"Failed to compute E Lz for orbit {i}\n{e}")

    return idx, cache_file, all_data, group_name


def callback(res):
    idx, cache_file, all_data, group_name = res

    logger.debug(f"Writing block {idx[0]}-{idx[-1]} to cache file")
    with h5py.File(cache_file, "r+") as f:
        grp = f[group_name]
        for k in all_data:
            grp[k][idx] = all_data[k]


def get_potential(fiducial_pot, eilers, **disk_pars):
    """
    Retrieve a MW potential model with fixed vcirc=229 at Rsun=8.275
    """
    from scipy.optimize import minimize

    eilers['xyz'] = np.zeros((len(eilers), 3))
    eilers['xyz'][:, 0] = eilers['R']
    def objfunc(vals):
        m = np.exp(vals[0])
        r_s = np.exp(vals[1])
        tmp_pot = fiducial_pot.replicate(
            disk=disk_pars,
            halo={'m': m, 'r_s': r_s}
        )
        test_v = tmp_pot.circular_velocity(
            eilers['xyz'].data.T,
        ).to_value(u.km / u.s)
        return np.sum((eilers['v_c'] - test_v)**2 / eilers['err']**2) + (r_s - 15.)**2 / 5**2

    x0 = [
        np.log(fiducial_pot['halo'].parameters["m"].to_value(u.Msun)),
        np.log(fiducial_pot['halo'].parameters["r_s"].to_value(u.kpc))
    ]

    res = minimize(objfunc, x0=x0, method="powell")
    if not res.success:
        raise RuntimeError(f"Failed to find potential for disk={disk_pars}")

    return fiducial_pot.replicate(
        disk=disk_pars,
        halo={'m': np.exp(res.x[0]), 'r_s': np.exp(res.x[1])}
    )


def main(
    pool,
    source_data_file,
    overwrite=False
):

    logger.debug(f"Starting file {source_data_file}...")

    cache_path = pathlib.Path(__file__).parent / "../cache"
    cache_path = cache_path.resolve().absolute()
    cache_path.mkdir(exist_ok=True)

    source_data_file = pathlib.Path(source_data_file).resolve()
    basename = source_data_file.name.split('.')[0]

    # Global parameters
    galcen_frame = coord.Galactocentric(
        galcen_distance=8.275 * u.kpc,
        galcen_v_sun=[8.4, 251.8, 8.4] * u.km/u.s
    )

    # TODO: these are hard-coded now, but should be specifiable through a config file or
    # something??
    fiducial_potential = gp.load('../data/potential.yml')

    # Load Eilers et al. 2019 circ. velocity curve
    eilers = at.Table.read('../data/elise/eilers_subset.csv')

    # TODO: should be specifiable somehow!
    # define grids of parameters to loop over (but keep vcirc fixed at 250 or whatever
    # at the solar circle)
    grid_Md = np.linspace(4, 9, 5) * 1e10
    grid_hz = np.linspace(0.3, 1.1, 5)

    # Load the source data table:
    g = GaiaData(at.QTable.read(source_data_file))

    c = g.get_skycoord()
    galcen = c.transform_to(galcen_frame)
    logger.debug("Data loaded...")

    Nstars = len(c)

    # Column metadata: map names to shapes
    meta = {
        "source_id": {
            "shape": (Nstars,),
            "dtype": g.source_id.dtype,
            "fillvalue": None,
        },
        "xyz": {"shape": (Nstars, 3), "unit": u.kpc},
        "vxyz": {"shape": (Nstars, 3), "unit": u.km / u.s},
        # Frequencies, actions, and angles computed with Sanders & Binney
        "freqs": {"shape": (Nstars, 3), "unit": u.rad / u.Gyr},
        "actions": {"shape": (Nstars, 3), "unit": u.kpc * u.km / u.s},
        "angles": {"shape": (Nstars, 3), "unit": u.rad},
        # Orbit parameters:
        # "z_max": {"shape": (Nstars,), "unit": u.kpc},
        # "r_per": {"shape": (Nstars,), "unit": u.kpc},
        # "r_apo": {"shape": (Nstars,), "unit": u.kpc},
        # "ecc": {"shape": (Nstars,), "unit": u.one},
        # "L": {"shape": (Nstars, 3), "unit": u.kpc * u.km / u.s},
        # "E": {"shape": (Nstars,), "unit": (u.km / u.s) ** 2},
    }

    cache_file = cache_path / f"pot-grid-{basename}.hdf5"
    logger.debug(f"Writing to cache file {cache_file}")

    if overwrite or not cache_file.exists():
        with h5py.File(cache_file, "w") as f:
            pass

    all_tasks = []
    for i, disk_m in enumerate(grid_Md):
        for j, disk_hz in enumerate(grid_hz):
            with h5py.File(cache_file, "r+") as f:
                group_name = f'{i:02d}_{j:02d}'

                if group_name not in f:
                    grp = f.create_group(group_name)
                else:
                    grp = f[group_name]

                if 'freqs' in grp.keys() and 'ecc' in grp.keys():
                    i1 = np.all(np.isnan(f["freqs"][:]), axis=1)
                    i2 = np.isnan(f["ecc"][:])
                    todo_idx = np.where(i1 & i2)[0]
                else:
                    todo_idx = np.arange(Nstars)

                grp.attrs['disk_m'] = disk_m
                grp.attrs['disk_h_z'] = disk_hz
                for name, info in meta.items():
                    if name in grp.keys():
                        continue

                    d = grp.create_dataset(
                        name,
                        shape=info["shape"],
                        dtype=info.get("dtype", "f8"),
                        fillvalue=info.get("fillvalue", np.nan),
                    )
                    if "unit" in info:
                        d.attrs["unit"] = str(info["unit"])

            try:
                pot = get_potential(fiducial_potential, eilers, m=disk_m, h_z=disk_hz)
            except RuntimeError:
                print(f"Failed to find pot for disk m={disk_m:.1e}, h_z={disk_hz:.2f}")
                continue

            n_batches = min(8 * max(1, pool.size - 1), len(todo_idx))
            tasks = batch_tasks(
                n_batches=n_batches,
                arr=todo_idx,
                args=(galcen, meta, pot, cache_file, group_name, g.source_id),
            )
            all_tasks.extend(tasks)

    for r in pool.map(worker, all_tasks, callback=callback):
        pass

    pool.close()
    sys.exit(0)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    from argparse import ArgumentParser

    parser = ArgumentParser(description="")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--nprocs",
        dest="n_procs",
        default=1,
        type=int,
        help="Number of processes (uses " "multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )

    parser.add_argument("-f", "--file", dest="source_data_file", required=True)
    parser.add_argument(
        "-o",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        default=False,
        action="store_true",
        help="Verbose mode.",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # deal with multiproc:
    if args.mpi:
        from schwimmbad.mpi import MPIPool

        Pool = MPIPool
        kw = dict()
    elif args.n_procs > 1:
        from schwimmbad import MultiPool

        Pool = MultiPool
        kw = dict(processes=args.n_procs)
    else:
        from schwimmbad import SerialPool

        Pool = SerialPool
        kw = dict()
    Pool = Pool
    Pool_kwargs = kw

    with threadpool_limits(limits=1, user_api="blas"):
        with Pool(**Pool_kwargs) as pool:
            main(
                pool,
                args.source_data_file,
                overwrite=args.overwrite
            )
