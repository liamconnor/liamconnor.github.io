import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.colors as mcolors
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import astropy.units as u
from pygdsm import GlobalSkyModel
import builtins
import h5py
from matplotlib.patches import Ellipse
# continuous catalog
# This can take a few seconds

# --- 1. Setup and Configuration ---

# Load or create the sky map
NSIDE = 32 # Using a slightly lower resoluation for faster animation rendering
#sky_map = create_synthetic_map(nside=NSIDE)
G = GlobalSkyModel()

# --- Telescope Location Setup ---
# Defaulting to Cambridge, MA. You can change this to your location.
# You can find your longitude/latitude from Google Maps.
# Longitude: West is negative, East is positive.
# Latitude: South is negative, North is positive.
location = EarthLocation(lon=-71.1097 * u.deg, lat=42.3736 * u.deg, height=10 * u.m)
# Current time in UTC
now_utc = Time.now()

# --- Animation Parameters ---
# We will simulate 24 hours of observation.
# The number of frames determines the smoothness and length of the animation.
N_FRAMES = 10 # Number of frames in the animation (reduced for testing fixed alpha mask)
TIME_SPAN_HOURS = 24 # The total duration of the observation in hours
time_deltas = np.linspace(0, TIME_SPAN_HOURS, N_FRAMES) * u.hour
observation_times = now_utc + time_deltas

def find_index(redshift, redshifts):
    """
    takes the target redshift and the redshifts array (e.g., z_continuous)
    returns index
    """
    index = np.where(np.round(redshifts,3)==np.round(redshift,3))[0][0]
    return index

# Define colormap globally
colors_blue = ['black', 'dodgerblue', 'lightskyblue', 'white']
n_bins = 1000
cmap_blue = mcolors.LinearSegmentedColormap.from_list('blue', colors_blue, N=n_bins)

def get_ralf_map(plot=False):
    name_continuous = '/Users/liamconnor/Downloads/Konietzka2025_DMmap_continuous_v1.hdf5'
    with h5py.File(name_continuous, 'r') as file:
        
        # load the data
        DM_continuous = file['DMvalues'][:]
        z_continuous  = file['redshifts'][:]

    # full-sky catalog
    # This can take a few seconds

    name_fullsky = '/Users/liamconnor/Downloads/Konietzka2025_DMmap_fullsky1_v1.hdf5'
    with h5py.File(name_fullsky, 'r') as file:
        
        # load the data
        DM_fullsky = file['DMvalues'][:]
        z_fullsky  = file['redshifts'][:]

    redshift_plot = 0.01
    idx_plot = find_index(redshift_plot, z_fullsky)
    DM_plot = DM_fullsky[idx_plot, :]

    if plot:
        hp.mollview(DM_plot, unit=r'DM [pc cm$^{-3}$]', min=int(0), 
                    max=int(100), xsize=800, cmap=cmap_blue, title='')
        plt.gca().set_title('')

        ax = plt.gca()
        ellipse = Ellipse(xy=(0, 0), width=4, height=2, edgecolor='black', 
                        facecolor='none', linewidth=1.5, transform=ax.transData)
        ax.add_patch(ellipse)

        plt.show()

    return DM_plot

#sky_map = G.generate(450 * u.MHz)
#sky_map = np.log10(hp.ud_grade(sky_map, nside_out=NSIDE))
sky_map = get_ralf_map(plot=False)
nside_ralf = hp.npix2nside(len(sky_map))
sky_map = hp.ud_grade(sky_map, nside_out=NSIDE)

# --- Yellow Star Persistence Parameters ---
yellow_star_frames_remaining = 0  # How many frames the yellow star should remain visible
yellow_star_position = None       # Store the position (theta, phi) of the yellow star
YELLOW_STAR_DURATION = 3.5         # How many frames the yellow star should last

# --- Red Star Parameters ---
RED_STAR_MEAN = 4                 # Mean number of red stars per frame (Poisson distribution)

# --- Overlay (Local Horizon) Parameters ---
# For a transit telescope, the "view" is what's above the horizon.
# A 90-degree radius from the zenith (straight up) represents the horizon.
OVERLAY_ALPHA = 0.4 # Transparency of the overlay for the part of the sky below the horizon

# --- 2. Create the Figure with Subplots ---
fig = plt.figure(figsize=(16, 10))

# Create subplots manually - healpy works better with simple subplot numbers
# We'll use a 1x2 layout where healpy takes the left side (121)
# Then we'll manually create the right side plots in a way that works

# Right side subplots - we'll position them manually after healpy
pass  # We'll create these after healpy to avoid conflicts

# Define the field of view - center at zenith (theta=0) and a reasonable radius  
# For zenith pointing: theta=0, phi=0 (North Pole in healpy coordinates)
center_vec = hp.ang2vec(0, 0)  # Zenith in healpy coordinates
radius_rad = np.pi/2  # 90 degrees - covers horizon to horizon

# --- 2. Define Plotting Parameters ---
# You can adjust these parameters to change the overlay.
CENTER_LON = 0      # Longitude of the circle's center in degrees (Galactic)
CENTER_LAT = 0      # Latitude of the circle's center in degrees (Galactic)
RADIUS_DEG = 80     # Radius of the circle in degrees
OVERLAY_COLOR = 'gray'
OVERLAY_ALPHA = 0.6 # Transparency (0=transparent, 1=opaque)

# --- 3. Create the Plot ---

# Figure already created above, don't create a new one

center_vec = hp.ang2vec(CENTER_LON, CENTER_LAT, lonlat=True)

# Find all pixels within the disc defined by the center and radius
npix = hp.nside2npix(NSIDE)
pixels_in_disc = hp.query_disc(NSIDE, center_vec, radius_rad, nest=False)
alpha_mask = np.zeros_like(sky_map) + 0.25
alpha_mask[pixels_in_disc] = 0.50

pixels_in_disc = hp.query_disc(NSIDE, center_vec, 0.9*radius_rad, nest=False)
alpha_mask[pixels_in_disc] = 0.75

# --- Bar Chart Setup ---
rate_casm_bar = 1            # Poisson mean per frame for CASM
rate_trad_bar = 0.1          # Poisson mean per frame for Traditional telescope
rng_bar = np.random.default_rng(42)
casm_bar_counts = [0]
trad_bar_counts = [0]
casm_bar_increments = rng_bar.poisson(rate_casm_bar, size=N_FRAMES)
trad_bar_increments = rng_bar.poisson(rate_trad_bar, size=N_FRAMES)

for i in range(N_FRAMES):
    casm_bar_counts.append(casm_bar_counts[-1] + casm_bar_increments[i])
    trad_bar_counts.append(trad_bar_counts[-1] + trad_bar_increments[i])

# Bar chart will be created dynamically in the update function

# --- Distance Histogram Setup ---
lambda_total_casm = 0.5   # expected CASM FRBs added per frame
lambda_total_trad = 5.0   # expected Traditional FRBs added per frame
r_min, r_max = 1.0, 10000.0
n_bins = 120
scale_casm, scale_trad = 50.0, 500.0
rng_hist = np.random.default_rng(12345)

# Distance bins
edges = np.linspace(r_min, r_max, n_bins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])

# Intensity shapes
shape_casm = centers**2 * np.exp(-centers / scale_casm)
shape_trad = centers**2 * np.exp(-centers / scale_trad)
shape_casm = shape_casm / shape_casm.sum() * lambda_total_casm
shape_trad = shape_trad / shape_trad.sum() * lambda_total_trad

# Precompute increments
inc_casm = rng_hist.poisson(shape_casm, size=(N_FRAMES, n_bins))
inc_trad = rng_hist.poisson(shape_trad, size=(N_FRAMES, n_bins))

# Distance histogram will be created dynamically in the update function

# Initialize histogram data
hist_casm = np.zeros(n_bins, dtype=int)
hist_trad = np.zeros(n_bins, dtype=int)

BRIGHT=False

# --- 3. Animation Update Function ---
# This function is called for each frame of the animation.
def update(frame_index):
    global BRIGHT, yellow_star_frames_remaining, yellow_star_position, hist_casm, hist_trad
    
    # Clear the healpy subplot before plotting new frame
    plt.subplot(1, 2, 1).clear()
    
    # Get the current time for this frame
    current_time = observation_times[frame_index]
    
    # --- Perform coordinate transformation from Galactic to local AltAz ---
    # For each pixel in our final AltAz map, we calculate where it points
    # in Galactic coordinates and sample the original sky map there.

    # 1. Define the pixel coordinates for the output map (in AltAz).
    npix = hp.nside2npix(NSIDE)
    theta_altaz, phi_altaz = hp.pix2ang(NSIDE, np.arange(npix), nest=False)
    
    # 2. Convert these healpy coordinates to astropy SkyCoord objects.
    # The AltAz frame requires the time and location of the observer.
    altaz_frame = AltAz(obstime=current_time, location=location)
    local_coords = SkyCoord(
        az=(phi_altaz * u.rad),
        alt=((np.pi/2.0 - theta_altaz) * u.rad),
        frame=altaz_frame
    )
    
    # 3. Transform these local coordinates to the Galactic coordinate system.
    galactic_coords = local_coords.transform_to('galactic')
    
    # 4. Convert transformed Galactic coordinates back to healpy (theta, phi).
    theta_galactic = np.pi/2.0 - galactic_coords.b.rad
    phi_galactic = galactic_coords.l.rad
    
    # 5. Find the pixel indices in the original Galactic map.
    source_pixels_galactic = hp.ang2pix(NSIDE, theta_galactic, 
                                        phi_galactic, nest=False)
    
    # 6. Create the new map by sampling the original sky_map.
    rotated_map = sky_map[source_pixels_galactic]
    
    # Plot the new map, which is now in the local (AltAz) coordinate system
    # Use healpy in left side of 1x2 layout
    hp.mollview(
        rotated_map,
        nest=False,
        min=0,
        max=int(100),
        cbar=False,
        alpha=alpha_mask,
        hold=False, # Required for alpha mask to work properly
        flip='astro',
        xsize=800, 
        cmap=cmap_blue,
        title='',
        notext=True,
        sub=121,  # Left side of 1x2 layout
        fig=fig.number
    )

    # Traditional telescope field of view (smaller circle)
    traditional_center_theta = np.pi/2.0  # Near horizon
    traditional_center_phi = -1.15  # Same longitude as text
    traditional_radius = 0.1  # Smaller radius than CASM
    
    # Create traditional telescope circle
    n_trad_points = 60
    trad_circle_theta = []
    trad_circle_phi = []
    
    # Simple circular field of view for traditional telescope
    for angle in np.linspace(0, 2*np.pi, n_trad_points):
        theta_offset = traditional_radius * np.cos(angle)
        phi_offset = traditional_radius * np.sin(angle)
        trad_circle_theta.append(traditional_center_theta + theta_offset)
        trad_circle_phi.append(traditional_center_phi + phi_offset)
    
    # Plot the traditional telescope circle
    hp.projplot(trad_circle_theta, trad_circle_phi, 'C1-', linewidth=2, alpha=0.8)

    fig.savefig(f'frame_{frame_index}.png')
    
    # Now create the right side subplots AFTER healpy
    ax_bar = plt.subplot(2, 2, 2)  # Top right
    ax_hist = plt.subplot(2, 2, 4)  # Bottom right
    
    # Set up bar chart
    ax_bar.clear()
    if frame_index < len(casm_bar_counts):
        current_casm_bar = casm_bar_counts[frame_index]
        current_trad_bar = trad_bar_counts[frame_index]
        
        categories = ["CASM", "Traditional FRB telescope"]
        bars = ax_bar.bar(categories, [current_casm_bar, current_trad_bar], color=["red", "goldenrod"])
        ax_bar.set_title("Local Universe FRBs", fontsize=14)
        ax_bar.set_ylim(0, builtins.max(casm_bar_counts[-1], trad_bar_counts[-1]) * 1.10)
        
        # Add labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
    
    # Set up distance histogram
    ax_hist.clear()
    if frame_index < N_FRAMES:
        hist_casm += inc_casm[frame_index]
        hist_trad += inc_trad[frame_index]
        
        ax_hist.plot(centers, hist_casm, drawstyle="steps-mid", 
                    label="CASM", color="red", lw=2)
        ax_hist.plot(centers, hist_trad, drawstyle="steps-mid", 
                    label="Traditional FRB telescope", 
                    color="goldenrod", lw=2)
        
        ax_hist.set_xscale("log")
        ax_hist.set_xlim(r_min*5, r_max)
        current_max = builtins.max(int(hist_casm.max()), int(hist_trad.max()))
        ax_hist.set_ylim(0, current_max * 1.10 if current_max > 0 else 1)
        ax_hist.set_xlabel("Distance (Mpc)", fontsize=10)
        ax_hist.set_ylabel("Number of FRBs", fontsize=10)
        ax_hist.legend(loc="upper left", fontsize=8)

    plt.gca().set_title('')

    ax = plt.gca()

    
    # Handle yellow star persistence
    if yellow_star_frames_remaining > 0:
        # Display the yellow star at its stored position
        hp.projplot(
                yellow_star_position[0],
                yellow_star_position[1],
                'r*', # Plot a yellow star
                markersize=17.5,
        )
        yellow_star_frames_remaining -= 1

    elif np.random.randint(0, 15) == 8:
        # Find pixels within field of view that have brightness > 50
        bright_pixels_mask = rotated_map[pixels_in_disc] > 50
        bright_pixels_in_disc = pixels_in_disc[bright_pixels_mask]
        
        if len(bright_pixels_in_disc) > 0:
            # Create a new yellow star at a random bright position
            selected_pixel = bright_pixels_in_disc[np.random.randint(0, len(bright_pixels_in_disc))]
            theta, phi = hp.pix2ang(NSIDE, selected_pixel, nest=False)
            yellow_star_position = (theta, phi)
            yellow_star_frames_remaining = YELLOW_STAR_DURATION - 1  # -1 because we're displaying it this frame
            # Add a single star marker at a random bright position within the field of view
            hp.projplot(
                    theta,
                    phi,
                    'r*', # Plot a red star
                    markersize=17.5,
                )

    # Generate multiple red stars using Poisson distribution
    N_red = np.random.poisson(RED_STAR_MEAN)
    
    # Find pixels within field of view that have brightness > 0
    bright_pixels_mask = rotated_map[pixels_in_disc] > 0
    bright_pixels_in_disc = pixels_in_disc[bright_pixels_mask]
    
    if len(bright_pixels_in_disc) > 0 and N_red > 0:
        # Generate N_red red stars at random bright positions
        for i in range(N_red):
            selected_pixel = bright_pixels_in_disc[np.random.randint(0, len(bright_pixels_in_disc))]
            theta, phi = hp.pix2ang(NSIDE, selected_pixel, nest=False)
            hp.projplot(
                    theta,
                    phi,
                    'y*', # Plot a red star
                    markersize=5,
                    alpha=0.90,
                )
                    
    # Draw a black circle around the field of view
    # Create points along the circle boundary
    n_circle_points = 360
    circle_theta = []
    circle_phi = []
    
    for angle in np.linspace(0, 2*np.pi, n_circle_points):
        # Create a vector in the plane perpendicular to center_vec
        # First, find two orthogonal vectors to center_vec
        if abs(center_vec[2]) < 0.9:
            v1 = np.cross(center_vec, [0, 0, 1])
        else:
            v1 = np.cross(center_vec, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(center_vec, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Point on the circle at the boundary
        circle_point = (center_vec * np.cos(radius_rad) + 
                       np.sin(radius_rad) * (v1 * np.cos(angle) + v2 * np.sin(angle)))
        
        # Convert to spherical coordinates
        theta_point = np.arccos(circle_point[2])
        phi_point = np.arctan2(circle_point[1], circle_point[0])
        if phi_point < 0:
            phi_point += 2*np.pi
            
        circle_theta.append(theta_point)
        circle_phi.append(phi_point)
    
    # Plot the circle
    hp.projplot(circle_theta, circle_phi, 'k-', linewidth=1.25, alpha=0.8)

    # Add "Below horizon" text to the shaded (low alpha) region 
    # Place text at a position outside the field of view circle
    # Choose a position at theta=2.5 (roughly 143 degrees from zenith), phi=0
    text_theta = np.pi/2.  # Outside the pi/2 radius circle
    text_phi = -2.25    # At longitude 0
    
    hp.projtext(text_theta, text_phi, "Below horizon", 
                fontsize=15, color='k', weight='bold',
                ha='center', va='center')
    
    hp.projtext(np.pi/2.0, -1.15, "Traditional\n telescope", 
                fontsize=8, color='C1', weight='bold',
                ha='center', va='center', )

    hp.projtext(np.pi/7.0, 0.0, "CASM-256", 
                fontsize=13.5, color='white', weight='bold',
                ha='center', va='center')

    # Bar chart and histogram updates are now handled above

    # Return value is not used when blit=False, but we return an empty tuple
    return ()

# --- 4. Create and Save the Animation ---
# Note: This process can be slow depending on your computer's performance.
print("Generating animation... This may take a few minutes.")

# Create the animation object
ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, blit=False)

# Save the animation as a GIF file
try:
    ani.save('combined_animation.gif', writer=PillowWriter(fps=12), dpi=100)
    print("\nAnimation saved successfully as 'combined_animation.gif'")
except Exception as e:
    print(f"\nCould not save animation. Error: {e}")
    print("Showing interactive plot instead.")
    plt.show()

# Create an animated bar chart showing cumulative Local Universe FRBs for
# CASM vs Traditional FRB telescope, where each frame updates counts by
# Poisson-distributed increments: CASM ~ Poisson(10), Traditional ~ Poisson(1).

import numpy as np
import matplotlib.pyplot as plt
import builtins
from matplotlib.animation import FuncAnimation, PillowWriter

def bar_chart():

    # Parameters
    n_frames = 200            # total frames in the animation
    rate_casm = 1            # Poisson mean per frame for CASM
    rate_trad = 0.1             # Poisson mean per frame for Traditional telescope

    # Random number generator (seed for reproducibility; remove or change if desired)
    rng = np.random.default_rng(42)

    # Storage for cumulative counts
    casm_counts = [0]
    trad_counts = [0]

    # Precompute Poisson increments for all frames for smoother animation
    casm_increments = rng.poisson(rate_casm, size=n_frames)
    trad_increments = rng.poisson(rate_trad, size=n_frames)

    for i in range(n_frames):
        casm_counts.append(casm_counts[-1] + casm_increments[i])
        trad_counts.append(trad_counts[-1] + trad_increments[i])

    # Set up the figure and bar chart
    fig, ax = plt.subplots(figsize=(6, 6))
    categories = ["CASM", "Traditional FRB telescope"]
    bars = ax.bar(categories, [0, 0], color=["red", "goldenrod"],)
    ax.set_title("Local Universe FRBs", fontsize=18)
    ax.set_ylim(0, builtins.max(casm_counts[-1], trad_counts[-1]) * 1.10)

    # Text labels above bars to show live counts
    labels = []
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        labels.append(ax.text(x, y, "0", ha="center", va="bottom"))

    # Update function for animation
    def update(frame):
        # frame runs from 0..n_frames; use precomputed cumulative counts
        current_casm = casm_counts[frame]
        current_trad = trad_counts[frame]

        # Update bar heights
        bars[0].set_height(current_casm)
        bars[1].set_height(current_trad)

        # Update labels
        labels[0].set_y(current_casm)
        labels[0].set_text(f"{current_casm}")
        labels[1].set_y(current_trad)
        labels[1].set_text(f"{current_trad}")

        return (*bars, *labels)

    anim = FuncAnimation(fig, update, frames=n_frames + 1, interval=50, blit=False)

    # Save as GIF using PillowWriter (avoids requiring ffmpeg)
    gif_path = "/Users/liamconnor/Downloads/frb_counts.gif"
    writer = PillowWriter(fps=20)
    anim.save(gif_path, writer=writer)

    gif_path

def distance_hist():

    # Animated histograms of FRB counts vs. Distance (Mpc), log10 x-axis.
    # Two instruments: CASM and Traditional FRB telescope.
    # Shapes per frame (Poisson means per bin):
    #   CASM:          ~ r^2 * exp(-r / 100 Mpc)
    #   Traditional:   ~ r^2 * exp(-r / 1000 Mpc)  # 1 Gpc
    #
    # We normalize each shape so that the *total* expected FRBs added per frame
    # is lambda_total_casm and lambda_total_trad, respectively, then draw Poisson
    # increments each frame and accumulate.
    #
    # No explicit colors are set (matplotlib will choose defaults).

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # ----------------------------- Parameters ----------------------------------
    n_frames = 300
    lambda_total_casm = 0.5   # expected CASM FRBs added per frame (sum over all bins)
    lambda_total_trad = 5.0    # expected Traditional FRBs added per frame (sum over all bins)

    r_min = 1.0        # Mpc (avoid zero so log scale behaves)
    r_max = 10000.0     # Mpc (2 Gpc)
    n_bins = 120       # number of distance bins

    scale_casm = 50.0     # Mpc
    scale_trad = 500.0    # Mpc (1 Gpc)

    rng = np.random.default_rng(12345)  # reproducibility; change/omit seed if desired
    # ---------------------------------------------------------------------------

    # Construct distance bins (linear spacing works fine; axis will be log-scaled)
    edges = np.linspace(r_min, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    # Define unnormalized per-bin intensity shapes
    shape_casm = centers**2 * np.exp(-centers / scale_casm)
    shape_trad = centers**2 * np.exp(-centers / scale_trad)

    # Convert shapes to Poisson means per bin per frame so that sums equal desired totals
    shape_casm = shape_casm / shape_casm.sum() * lambda_total_casm
    shape_trad = shape_trad / shape_trad.sum() * lambda_total_trad

    # Precompute Poisson increments for efficiency: arrays [n_frames, n_bins]
    inc_casm = rng.poisson(shape_casm, size=(n_frames, n_bins))
    inc_trad = rng.poisson(shape_trad, size=(n_frames, n_bins))

    # ----------------------------- Figure setup --------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use step-style lines to represent histograms cleanly
    (line_casm,) = ax.plot([], [], drawstyle="steps-mid", 
                           label="CASM", color="red", lw=3)
    (line_trad,) = ax.plot([], [], drawstyle="steps-mid", 
                           label="Traditional FRB telescope", 
                           color="goldenrod", lw=3)

    ax.set_xscale("log")
    ax.set_xlim(r_min*5, r_max)
    ax.set_ylim(0, 1)  # will autoscale upward on first frame
    ax.set_xlabel("Distance (Mpc)")
    ax.set_ylabel("Number of FRBs")
    ax.legend(loc="upper left")

    # define these BEFORE defining update()
    hist_casm = np.zeros(n_bins, dtype=int)
    hist_trad = np.zeros(n_bins, dtype=int)

    def update(frame, hist_casm=hist_casm, hist_trad=hist_trad):
        # Poisson increments for this frame
        hist_casm += inc_casm[frame]
        hist_trad += inc_trad[frame]

        line_casm.set_data(centers, hist_casm)
        line_trad.set_data(centers, hist_trad)
 
        current_max = builtins.max(int(hist_casm.max()), int(hist_trad.max()))
        ymin, ymax = ax.get_ylim()
        if current_max > 0.95 * ymax:
            ax.set_ylim(0, current_max * 1.10)

        return (line_casm, line_trad)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)

    # Save as GIF
    gif_path = "/Users/liamconnor/Downloads/frb_distance_hist.gif"
    PillowWriter(fps=20).setup(fig, gif_path, dpi=100)
    anim.save(gif_path, writer=PillowWriter(fps=20))

# bar_chart()  # Now integrated into main animation
# distance_hist()  # Now integrated into main animation