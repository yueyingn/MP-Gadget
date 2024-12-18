#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

#include "utils.h"

#include "allvars.h"
#include "walltime.h"
#include "gravity.h"
#include "density.h"
#include "domain.h"
#include "run.h"
#include "init.h"
#include "cooling.h"
#include "checkpoint.h"
#include "petaio.h"
#include "petapm.h"
#include "timestep.h"
#include "drift.h"
#include "forcetree.h"
#include "blackhole.h"
#include "hydra.h"
#include "sfr_eff.h"
#include "metal_return.h"
#include "fdm.h"
#include "slotsmanager.h"
#include "hci.h"
#include "fof.h"
#include "cooling_qso_lightup.h"
#include "lightcone.h"
#include "timefac.h"

/* stats.c only used here */
void energy_statistics(FILE * FdEnergy, const double Time,  struct part_manager_type * PartManager);
/*!< file handle for energy.txt log-file. */
static FILE * FdEnergy;
static FILE  *FdCPU;    /*!< file handle for cpu.txt log-file. */
static FILE *FdSfr;     /*!< file handle for sfr.txt log-file. */
static FILE *FdBlackHoles;  /*!< file handle for blackholes.txt log-file. */
static FILE *FdBlackholeDetails;  /*!< file handle for BlackholeDetails binary file. */

static struct ClockTable Clocks;

/*! \file run.c
 *  \brief  iterates over timesteps, main loop
 */
static void write_cpu_log(int NumCurrentTiStep, FILE * FdCPU);

/* Updates the global storing the current random offset of the particles,
 * and stores the relative offset from the last random offset in rel_random_shift*/
static void update_random_offset(double * rel_random_shift);
static void set_units();

static void
open_outputfiles(int RestartSnapNum);

static void
close_outputfiles(void);

/*! This function performs the initial set-up of the simulation. First, the
 *  parameterfile is set, then routines for setting units, reading
 *  ICs/restart-files are called, auxialiary memory is allocated, etc.
 */
int begrun(int RestartFlag, int RestartSnapNum)
{
    if(RestartFlag == 1) {
        RestartSnapNum = find_last_snapnum(All.OutputDir);
        message(0, "Last Snapshot number is %d.\n", RestartSnapNum);
    }

    hci_init(HCI_DEFAULT_MANAGER, All.OutputDir, All.TimeLimitCPU, All.AutoSnapshotTime, All.SnapshotWithFOF);

    petapm_module_init(omp_get_max_threads());
    petaio_init();
    walltime_init(&Clocks);

    petaio_read_header(RestartSnapNum);

    slots_init(All.SlotsIncreaseFactor * PartManager->MaxPart, SlotsManager);
    /* Enable the slots: stars and BHs are allocated if there are some,
     * or if some will form*/
    if(All.NTotalInit[0] > 0)
        slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    if(All.StarformationOn || All.NTotalInit[4] > 0)
        slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    if(All.BlackHoleOn || All.NTotalInit[5] > 0)
        slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    if(All.FdmOn || All.NTotalInit[1] > 0)
        slots_set_enabled(1, sizeof(struct fdm_particle_data), SlotsManager);

    set_units();

#ifdef DEBUG
    char * pidfile = fastpm_strdup_printf("%s/%s", All.OutputDir, "PIDs.txt");

    MPIU_write_pids(pidfile);
    myfree(pidfile);

    enable_core_dumps_and_fpu_exceptions();
#endif

    init_forcetree_params(All.FastParticleType);

    init_cooling_and_star_formation(All.CoolingOn);

    gravshort_set_softenings(All.MeanSeparation[1]);
    gravshort_fill_ntab(All.ShortRangeForceWindowType, All.Asmth);

    set_random_numbers(All.RandomSeed);

    if(All.LightconeOn)
        lightcone_init(&All.CP, All.Time);
    return RestartSnapNum;
}

/* Small function to decide - collectively - whether to use pairwise gravity this step*/
static int
use_pairwise_gravity(ActiveParticles * Act, struct part_manager_type * PartManager)
{
    /* Find total number of active particles*/
    int64_t total_active, total_particle;
    MPI_Allreduce(&Act->NumActiveParticle, &total_active, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&PartManager->NumPart, &total_particle, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* Since the pairwise step is O(N^2) and tree is O(NlogN) we should scale the condition like O(N)*/
    return total_active < All.PairwiseActiveFraction * total_particle;
}

/*! This routine contains the main simulation loop that iterates over
 * single timesteps. The loop terminates when the cpu-time limit is
 * reached, when a `stop' file is found in the output directory, or
 * when the simulation ends because we arrived at TimeMax.
 */
void
run(int RestartSnapNum)
{
    /*Number of timesteps performed this run*/
    int NumCurrentTiStep = 0;
    /*Is gas physics enabled?*/
    int GasEnabled = All.NTotalInit[0] > 0;

    int SnapshotFileCount = RestartSnapNum;
    PetaPM pm = {0};
    gravpm_init_periodic(&pm, All.BoxSize, All.Asmth, All.Nmesh, All.G);

    DomainDecomp ddecomp[1] = {0};

    /* ... read initial model and initialise the times*/
    DriftKickTimes times = init_driftkicktime(init(RestartSnapNum, ddecomp));

    /* Stored scale factor of the next black hole seeding check*/
    double TimeNextSeedingCheck = All.Time;

    walltime_measure("/Misc");

    open_outputfiles(RestartSnapNum);

    write_cpu_log(NumCurrentTiStep, FdCPU); /* produce some CPU usage info */
    
    write_checkpoint(99, 1, 0, All.Time, All.OutputDir, All.SnapshotFileBase, All.OutputDebugFields);

    close_outputfiles();
}


void write_cpu_log(int NumCurrentTiStep, FILE * FdCPU)
{
    walltime_summary(0, MPI_COMM_WORLD);

    if(FdCPU)
    {
        int NTask;
        MPI_Comm_size(MPI_COMM_WORLD, &NTask);
        fprintf(FdCPU, "Step %d, Time: %g, MPIs: %d Threads: %d Elapsed: %g\n", NumCurrentTiStep, All.Time, NTask, omp_get_max_threads(), Clocks.ElapsedTime);
        walltime_report(FdCPU, 0, MPI_COMM_WORLD);
        fflush(FdCPU);
    }
}

/* We operate in a situation where the particles are in a coordinate frame
 * offset slightly from the ICs (to avoid correlated tree errors).
 * This function updates the global variable containing that offset, and
 * stores the relative shift from the last offset in the rel_random_shift output
 * array. */
static void
update_random_offset(double * rel_random_shift)
{
    int i;
    for (i = 0; i < 3; i++) {
        /* Note random number table is duplicated across processors*/
        double rr = get_random_number(i);
        /* Upstream Gadget uses a random fraction of the box, but since all we need
         * is to adjust the tree openings, and the tree force is zero anyway on the
         * scale of a few PM grid cells, this seems enough.*/
        rr *= All.RandomParticleOffset * All.BoxSize / All.Nmesh;
        /* Subtract the old random shift first.*/
        rel_random_shift[i] = rr - PartManager->CurrentParticleOffset[i];
        PartManager->CurrentParticleOffset[i] = rr;
    }
    message(0, "Internal particle offset is now %g %g %g\n", PartManager->CurrentParticleOffset[0], PartManager->CurrentParticleOffset[1], PartManager->CurrentParticleOffset[2]);
#ifdef DEBUG
    /* Check explicitly that the vector is the same on all processors*/
    double test_random_shift[3] = {0};
    for (i = 0; i < 3; i++)
        test_random_shift[i] = PartManager->CurrentParticleOffset[i];
    MPI_Bcast(test_random_shift, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (i = 0; i < 3; i++)
        if(test_random_shift[i] != PartManager->CurrentParticleOffset[i])
            endrun(44, "Random shift %d is %g != %g on task 0!\n", i, test_random_shift[i], PartManager->CurrentParticleOffset[i]);
#endif
}

/*!  This function opens various log-files that report on the status and
 *   performance of the simulstion. On restart from restart-files
 *   (start-option 1), the code will append to these files.
 */
static void
open_outputfiles(int RestartSnapNum)
{
    const char mode[3]="a+";
    char * buf;
    char * postfix;
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    FdCPU = NULL;
    FdEnergy = NULL;
    FdBlackHoles = NULL;
    FdSfr = NULL;
    FdBlackholeDetails = NULL;

    if(RestartSnapNum != -1) {
        postfix = fastpm_strdup_printf("-R%03d", RestartSnapNum);
    } else {
        postfix = fastpm_strdup_printf("%s", "");
    }

    /* all the processors write to separate files*/
    if(All.BlackHoleOn && All.WriteBlackHoleDetails){
        buf = fastpm_strdup_printf("%s/%s%s/%06X", All.OutputDir,"BlackholeDetails",postfix,ThisTask);
        fastpm_path_ensure_dirname(buf);
        if(!(FdBlackholeDetails = fopen(buf,"a")))
            endrun(1, "Failed to open blackhole detail %s\n", buf);
        myfree(buf);
    }

    /* only the root processors writes to the log files */
    if(ThisTask != 0) {
        return;
    }

    buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, All.CpuFile, postfix);
    fastpm_path_ensure_dirname(buf);
    if(!(FdCPU = fopen(buf, mode)))
        endrun(1, "error in opening file '%s'\n", buf);
    myfree(buf);

    if(All.OutputEnergyDebug) {
        buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, All.EnergyFile, postfix);
        fastpm_path_ensure_dirname(buf);
        if(!(FdEnergy = fopen(buf, mode)))
            endrun(1, "error in opening file '%s'\n", buf);
        myfree(buf);
    }

    if(All.StarformationOn) {
        buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, "sfr.txt", postfix);
        fastpm_path_ensure_dirname(buf);
        if(!(FdSfr = fopen(buf, mode)))
            endrun(1, "error in opening file '%s'\n", buf);
        myfree(buf);
    }

    if(All.BlackHoleOn) {
        buf = fastpm_strdup_printf("%s/%s%s", All.OutputDir, "blackholes.txt", postfix);
        fastpm_path_ensure_dirname(buf);
        if(!(FdBlackHoles = fopen(buf, mode)))
            endrun(1, "error in opening file '%s'\n", buf);
        myfree(buf);
    }
}


/*!  This function closes the global log-files.
*/
static void
close_outputfiles(void)
{
    if(FdCPU)
        fclose(FdCPU);
    if(FdEnergy)
        fclose(FdEnergy);
    if(FdSfr)
        fclose(FdSfr);
    if(FdBlackHoles)
        fclose(FdBlackHoles);
    if(FdBlackholeDetails)
        fclose(FdBlackholeDetails);
}

/*! Computes conversion factors between internal code units and the
 *  cgs-system.
 */
static void
set_units(void)
{
    All.UnitTime_in_s = All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s;
    All.UnitTime_in_Megayears = All.UnitTime_in_s / SEC_PER_MEGAYEAR;

    All.G = GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);

    All.UnitDensity_in_cgs = All.UnitMass_in_g / pow(All.UnitLength_in_cm, 3);
    All.UnitEnergy_in_cgs = All.UnitMass_in_g * pow(All.UnitLength_in_cm, 2) / pow(All.UnitTime_in_s, 2);

    /* convert some physical input parameters to internal units */

    All.CP.Hubble = HUBBLE * All.UnitTime_in_s;
    All.CP.UnitTime_in_s = All.UnitTime_in_s;

    init_cosmology(&All.CP, All.TimeIC);
    /* Detect cosmologies that are likely to be typos in the parameter files*/
    if(All.CP.HubbleParam < 0.1 || All.CP.HubbleParam > 10 ||
        All.CP.OmegaLambda < 0 || All.CP.OmegaBaryon < 0 || All.CP.OmegaG < 0 || All.CP.OmegaCDM < 0)
        endrun(5, "Bad cosmology: H0 = %g OL = %g Ob = %g Og = %g Ocdm = %g\n",
               All.CP.HubbleParam, All.CP.OmegaLambda, All.CP.OmegaBaryon, All.CP.OmegaCDM);


    if(All.InitGasTemp < 0)
        All.InitGasTemp = All.CP.CMBTemperature / All.TimeInit;
    /*Initialise the hybrid neutrinos, after Omega_nu*/
    if(All.HybridNeutrinosOn)
        init_hybrid_nu(&All.CP.ONu.hybnu, All.CP.MNu, All.HybridVcrit, LIGHTCGS/1e5, All.HybridNuPartTime, All.CP.ONu.kBtnu);

    message(0, "Hubble (internal units) = %g\n", All.CP.Hubble);
    message(0, "G (internal units) = %g\n", All.G);
    message(0, "UnitLength_in_cm = %g \n", All.UnitLength_in_cm);
    message(0, "UnitMass_in_g = %g \n", All.UnitMass_in_g);
    message(0, "UnitTime_in_s = %g \n", All.UnitTime_in_s);
    message(0, "UnitVelocity_in_cm_per_s = %g \n", All.UnitVelocity_in_cm_per_s);
    message(0, "UnitDensity_in_cgs = %g \n", All.UnitDensity_in_cgs);
    message(0, "UnitEnergy_in_cgs = %g \n", All.UnitEnergy_in_cgs);
    message(0, "Dark energy model: OmegaL = %g OmegaFLD = %g\n",All.CP.OmegaLambda, All.CP.Omega_fld);
    message(0, "Photon density OmegaG = %g\n",All.CP.OmegaG);
    if(!All.MassiveNuLinRespOn)
        message(0, "Massless Neutrino density OmegaNu0 = %g\n",get_omega_nu(&All.CP.ONu, 1));
    message(0, "Curvature density OmegaK = %g\n",All.CP.OmegaK);
    if(All.CP.RadiationOn) {
        /* note that this value is inaccurate if there is massive neutrino. */
        double OmegaTot = All.CP.OmegaG + All.CP.OmegaK + All.CP.Omega0 + All.CP.OmegaLambda;
        if(!All.MassiveNuLinRespOn)
            OmegaTot += get_omega_nu(&All.CP.ONu, 1);
        message(0, "Radiation is enabled in Hubble(a). "
               "Following CAMB convention: Omega_Tot - 1 = %g\n", OmegaTot - 1);
    }
    message(0, "\n");
}
