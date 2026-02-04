"""Visualization module for METRIC ETa pipeline.

This module provides classes for creating visualization products
including spatial maps, time series, histograms, and energy balance plots.
"""

from typing import Dict, Optional, List, Any, Union, Tuple
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from datetime import datetime
from pathlib import Path

from metric_et.core.datacube import DataCube


class Visualization:
    """Visualization class for METRIC ETa products.
    
    This class provides methods for creating various visualization products
    from the METRIC processing pipeline.
    
    Attributes:
        output_dir: Base output directory for visualization files
        figure_dpi: DPI for output figures
        default_colormap: Default colormap for ET products
    """
    
    # Default colormaps for different products
    COLORMAPS = {
        'ET': 'viridis',
        'ETa': 'YlGn',
        'ETrF': 'RdYlGn',
        'LE': 'plasma',
        'Rn': 'RdBu_r',
        'G': 'Oranges',
        'H': 'YlOrRd',
        'NDVI': 'YlGn',
        'LST': 'turbo'
    }
    
    # Colorbar labels
    LABELS = {
        'ET': 'ET (mm/day)',
        'ETa': 'ETa (mm/day)',
        'ETrF': 'ETrF (dimensionless)',
        'LE': 'Latent Heat (W/m²)',
        'Rn': 'Net Radiation (W/m²)',
        'G': 'Soil Heat Flux (W/m²)',
        'H': 'Sensible Heat Flux (W/m²)',
        'NDVI': 'NDVI',
        'LST': 'Surface Temperature (K)'
    }
    
    def __init__(
        self,
        output_dir: str = ".",
        figure_dpi: int = 150,
        default_colormap: str = 'viridis'
    ):
        """Initialize the Visualization class.
        
        Args:
            output_dir: Base output directory for visualization files
            figure_dpi: DPI for output figures
            default_colormap: Default colormap for ET products
        """
        self.output_dir = Path(output_dir)
        self.figure_dpi = figure_dpi
        self.default_colormap = default_colormap
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_colormap(self, product: str) -> str:
        """Get colormap for a product.
        
        Args:
            product: Product name
            
        Returns:
            Colormap name
        """
        return self.COLORMAPS.get(product, self.default_colormap)
    
    def _get_label(self, product: str) -> str:
        """Get colorbar label for a product.
        
        Args:
            product: Product name
            
        Returns:
            Label string
        """
        return self.LABELS.get(product, f'{product}')
    
    def _save_figure(
        self,
        fig: plt.Figure,
        output_path: Optional[str],
        show: bool = False
    ) -> Optional[str]:
        """Save figure to file and optionally display.
        
        Args:
            fig: Matplotlib figure object
            output_path: Output file path (None to auto-generate)
            show: Whether to display the figure
            
        Returns:
            Output file path if saved
        """
        if show:
            plt.show()
        
        if output_path:
            fig.savefig(
                output_path,
                dpi=self.figure_dpi,
                bbox_inches='tight',
                facecolor='white'
            )
            return output_path
        
        return None
    
    def plot_et_map(
        self,
        data: Union[np.ndarray, xr.DataArray],
        cube: DataCube,
        title: str = "Evapotranspiration Map",
        product: str = 'ETa',
        colormap: Optional[str] = None,
        output_path: Optional[str] = None,
        show: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        overlay_contours: bool = False
    ) -> Optional[str]:
        """Create spatial ETa map.
        
        Args:
            data: 2D numpy array or xarray.DataArray
            cube: DataCube containing CRS and transform information
            title: Plot title
            product: Product type for colormap selection
            colormap: Override colormap (auto-selected if None)
            output_path: Output file path
            show: Whether to display the plot
            vmin: Minimum value for color scaling
            vmax: Maximum value for color scaling
            overlay_contours: Whether to add contours
            
        Returns:
            Output file path if saved
        """
        # Convert to numpy array
        if isinstance(data, xr.DataArray):
            data = data.values
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get colormap and label
        cmap = colormap or self._get_colormap(product)
        label = self._get_label(product)
        
        # Create masked array for NaN handling
        data_masked = np.ma.masked_invalid(data)
        
        # Determine vmin/vauto
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            if vmin is None:
                vmin = np.nanmin(valid_data)
            if vmax is None:
                vmax = np.nanmax(valid_data)
        
        # Plot data
        im = ax.imshow(
            data_masked,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='upper'
        )
        
        # Add contours if requested
        if overlay_contours and len(valid_data) > 0:
            levels = np.linspace(vmin, vmax, 10)
            ax.contour(data, levels=levels, colors='black', linewidths=0.5, alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(label)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        
        # Add scene info if available
        scene_id = cube.metadata.get('scene_id', '')
        if scene_id:
            ax.text(
                0.02, 0.98, f'Scene: {scene_id}',
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        plt.tight_layout()
        
        return self._save_figure(fig, output_path, show)
    
    def plot_et_timeseries(
        self,
        et_data_list: List[np.ndarray],
        dates: List[datetime],
        locations: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        show: bool = False,
        title: str = "ET Time Series",
        ylabel: str = "ETa (mm/day)",
        error_bars: bool = False
    ) -> Optional[str]:
        """Create ETa time series plot.
        
        Args:
            et_data_list: List of ET arrays for each date
            dates: List of datetime objects
            locations: Optional list of location names
            output_path: Output file path
            show: Whether to display the plot
            title: Plot title
            ylabel: Y-axis label
            error_bars: Whether to add error bars
            
        Returns:
            Output file path if saved
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Compute mean ET for each date
        et_means = [np.nanmean(et) for et in et_data_list]
        
        # Compute std for error bars
        if error_bars:
            et_stds = [np.nanstd(et) for et in et_data_list]
        
        # Convert dates to matplotlib format
        mpl_dates = [plt.matplotlib.dates.date2num(d) for d in dates]
        
        # Plot
        if error_bars:
            ax.errorbar(
                mpl_dates, et_means, yerr=et_stds,
                fmt='o-', capsize=3, capthick=1, linewidth=1.5,
                markersize=6, color='steelblue'
            )
        else:
            ax.plot(mpl_dates, et_means, 'o-', linewidth=1.5, markersize=6)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter('%Y-%m-%d')
        )
        ax.xaxis.set_major_locator(
            plt.matplotlib.dates.WeekdayLocator(interval=1)
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add location labels if provided
        if locations:
            ax.legend(locations, loc='upper right')
        
        plt.tight_layout()
        
        return self._save_figure(fig, output_path, show)
    
    def plot_et_histogram(
        self,
        data: Union[np.ndarray, xr.DataArray],
        bins: int = 50,
        output_path: Optional[str] = None,
        show: bool = False,
        title: str = "ET Distribution",
        xlabel: str = "ET (mm/day)",
        overlay_stats: bool = True
    ) -> Optional[str]:
        """Create ETa distribution histogram.
        
        Args:
            data: 2D numpy array or xarray.DataArray
            bins: Number of histogram bins
            output_path: Output file path
            show: Whether to display the plot
            title: Plot title
            xlabel: X-axis label
            overlay_stats: Whether to overlay mean/median lines
            
        Returns:
            Output file path if saved
        """
        # Convert to numpy array
        if isinstance(data, xr.DataArray):
            data = data.values
        
        # Remove NaN values
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            raise ValueError("No valid data for histogram")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        n, bins_edges, patches = ax.hist(
            valid_data, bins=bins, color='steelblue',
            edgecolor='white', alpha=0.7
        )
        
        # Overlay statistics
        if overlay_stats:
            mean_val = np.mean(valid_data)
            median_val = np.median(valid_data)
            std_val = np.std(valid_data)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-', linewidth=2,
                      label=f'Median: {median_val:.2f}')
            
            ax.legend(loc='upper right')
            
            # Add stats text box
            stats_text = (f'Mean: {mean_val:.2f}\n'
                         f'Median: {median_val:.2f}\n'
                         f'Std: {std_val:.2f}\n'
                         f'N: {len(valid_data):,}')
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
            )
        
        # Labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        return self._save_figure(fig, output_path, show)
    
    def plot_energy_balance(
        self,
        Rn: Union[np.ndarray, xr.DataArray],
        G: Union[np.ndarray, xr.DataArray],
        H: Union[np.ndarray, xr.DataArray],
        LE: Union[np.ndarray, xr.DataArray],
        output_path: Optional[str] = None,
        show: bool = False,
        title: str = "Energy Balance Components",
        stacked: bool = True
    ) -> Optional[str]:
        """Create energy balance bar chart.
        
        Args:
            Rn: Net radiation array (W/m²)
            G: Soil heat flux array (W/m²)
            H: Sensible heat flux array (W/m²)
            LE: Latent heat flux array (W/m²)
            output_path: Output file path
            show: Whether to display the plot
            title: Plot title
            stacked: Whether to use stacked bars
            
        Returns:
            Output file path if saved
        """
        # Convert to numpy arrays if needed
        if isinstance(Rn, xr.DataArray):
            Rn = Rn.values
        if isinstance(G, xr.DataArray):
            G = G.values
        if isinstance(H, xr.DataArray):
            H = H.values
        if isinstance(LE, xr.DataArray):
            LE = LE.values
        
        # Compute mean values
        Rn_mean = np.nanmean(Rn)
        G_mean = np.nanmean(G)
        H_mean = np.nanmean(H)
        LE_mean = np.nanmean(LE)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left panel: Stacked bar chart
        ax1 = axes[0]
        if stacked:
            labels = ['Mean Scene Values']
            Rn_vals = [Rn_mean]
            G_vals = [G_mean]
            H_vals = [H_mean]
            LE_vals = [LE_mean]
            
            # Create stacked bars
            bottom = np.zeros(1)
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            components = [('G', G_vals), ('H', H_vals), ('LE', LE_vals)]
            
            ax1.bar(labels, Rn_vals, label='Rn', color='#2E86AB', alpha=0.8)
            
            # Check energy balance closure
            residual = Rn_mean - (G_mean + H_mean + LE_mean)
            if residual < 0:
                ax1.bar(labels, [residual], bottom=[Rn_mean],
                       label='Residual', color='gray', alpha=0.5)
            
            ax1.axhline(Rn_mean, color='black', linestyle='--',
                       linewidth=1, label=f'Rn = {Rn_mean:.1f}')
            
            ax1.set_ylabel('Energy Flux (W/m²)')
            ax1.set_title('Energy Components (Stacked)')
            ax1.legend()
        else:
            components = [('Rn', Rn_mean), ('G', G_mean),
                         ('H', H_mean), ('LE', LE_mean)]
            names = [c[0] for c in components]
            values = [c[1] for c in components]
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            
            bars = ax1.bar(names, values, color=colors, alpha=0.8)
            ax1.set_ylabel('Energy Flux (W/m²)')
            ax1.set_title('Energy Components')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Right panel: Pie chart of relative proportions
        ax2 = axes[1]
        
        # Calculate proportions (excluding Rn)
        total = G_mean + H_mean + LE_mean
        if total > 0:
            proportions = [G_mean/total * 100, H_mean/total * 100, LE_mean/total * 100]
            labels = [f'G ({proportions[0]:.1f}%)',
                     f'H ({proportions[1]:.1f}%)',
                     f'LE ({proportions[2]:.1f}%)']
            colors_pie = ['#A23B72', '#F18F01', '#C73E1D']
            
            wedges, texts, autotexts = ax2.pie(
                [G_mean, H_mean, LE_mean],
                labels=labels,
                colors=colors_pie,
                autopct='',
                startangle=90
            )
            ax2.set_title('Proportion of Energy Balance')
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return self._save_figure(fig, output_path, show)
    
    def plot_rgb_composite(
        self,
        cube: DataCube,
        output_path: Optional[str] = None,
        show: bool = False,
        title: str = "True Color RGB Composite",
        contrast_stretch: bool = True,
        brightness: float = 1.0
    ) -> Optional[str]:
        """Create true-color RGB composite.
        
        Args:
            cube: DataCube containing blue, green, red bands
            output_path: Output file path
            show: Whether to display the plot
            title: Plot title
            contrast_stretch: Whether to apply contrast stretching
            brightness: Brightness multiplier
            
        Returns:
            Output file path if saved
        """
        # Check for required bands
        required_bands = ['blue', 'green', 'red']
        missing = [b for b in required_bands if b not in cube.data]
        if missing:
            raise ValueError(f"Missing bands for RGB composite: {missing}")
        
        # Get bands
        blue = cube.data['blue'].values.astype(np.float32)
        green = cube.data['green'].values.astype(np.float32)
        red = cube.data['red'].values.astype(np.float32)
        
        # Apply contrast stretching
        if contrast_stretch:
            def stretch(arr, p=2):
                """Percentile-based contrast stretch."""
                low, high = np.nanpercentile(arr, (p, 100-p))
                return np.clip((arr - low) / (high - low) * 255, 0, 255)
            blue = stretch(blue)
            green = stretch(green)
            red = stretch(red)
        
        # Apply brightness
        blue = np.clip(blue * brightness, 0, 255)
        green = np.clip(green * brightness, 0, 255)
        red = np.clip(red * brightness, 0, 255)
        
        # Stack RGB
        rgb = np.dstack([red, green, blue]).astype(np.uint8)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.imshow(rgb)
        ax.set_title(title)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        
        plt.tight_layout()
        
        return self._save_figure(fig, output_path, show)
    
    def create_summary_figure(
        self,
        cube: DataCube,
        output_path: Optional[str] = None,
        show: bool = False,
        calibration_result: Optional[Any] = None,
        anchor_result: Optional[Any] = None  # NEW: AnchorPixelsResult with clusters
    ) -> Optional[str]:
        """Create summary figure with multiple panels.

        Args:
            cube: DataCube containing ET products
            output_path: Output file path
            show: Whether to display the plot
            calibration_result: Calibration result containing anchor pixel coordinates
            anchor_result: AnchorPixelsResult containing cold_cluster and hot_cluster

        Returns:
            Output file path if saved
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Panel 1: RGB Composite
        ax1 = axes[0, 0]
        try:
            if all(b in cube.data for b in ['blue', 'green', 'red']):
                blue = cube.data['blue'].values.astype(np.float32)
                green = cube.data['green'].values.astype(np.float32)
                red = cube.data['red'].values.astype(np.float32)
                
                # Simple contrast stretch
                def stretch(arr):
                    p2, p98 = np.nanpercentile(arr, (2, 98))
                    result = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255)
                    return np.nan_to_num(result, nan=0)

                rgb = np.dstack([
                    stretch(red),
                    stretch(green),
                    stretch(blue)
                ]).astype(np.uint8)
                
                ax1.imshow(rgb)
                ax1.set_title('RGB Composite')
            else:
                ax1.text(0.5, 0.5, 'RGB bands not available',
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('RGB Composite (Not Available)')
        except Exception:
            ax1.text(0.5, 0.5, 'Error creating RGB',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('RGB Composite (Error)')
        
        ax1.set_xlabel('Pixel X')
        ax1.set_ylabel('Pixel Y')

        # Add anchor pixel markers from anchor_result (cluster-based)
        if anchor_result is not None:
            import logging
            logger = logging.getLogger(__name__)

            # Plot cold cluster
            if hasattr(anchor_result, 'cold_cluster') and anchor_result.cold_cluster is not None:
                cold_pixels = anchor_result.cold_cluster.pixel_indices
                logger.info(f"Visualization: Cold cluster has {len(cold_pixels)} pixels")
                cold_xs = [p[1] for p in cold_pixels]  # x is second element
                cold_ys = [p[0] for p in cold_pixels]  # y is first element

                # Plot all cluster pixels with more visible markers
                ax1.scatter(cold_xs, cold_ys, c='cyan', marker='o', s=20,
                           edgecolors='blue', linewidth=1, alpha=0.8,
                           label=f'Cold Cluster ({len(cold_pixels)} pixels)')

                # Highlight the best cold pixel (first in list)
                if len(cold_pixels) > 0:
                    best_cold_x = cold_pixels[0][1]
                    best_cold_y = cold_pixels[0][0]
                    ax1.scatter(best_cold_x, best_cold_y, c='blue', marker='o', s=150,
                               edgecolors='white', linewidth=3, alpha=1.0,
                               label=f'Cold Best (Ts={anchor_result.cold_cluster.ts_median:.1f}K)')

            # Plot hot cluster
            if hasattr(anchor_result, 'hot_cluster') and anchor_result.hot_cluster is not None:
                hot_pixels = anchor_result.hot_cluster.pixel_indices
                logger.info(f"Visualization: Hot cluster has {len(hot_pixels)} pixels")
                hot_xs = [p[1] for p in hot_pixels]
                hot_ys = [p[0] for p in hot_pixels]

                ax1.scatter(hot_xs, hot_ys, c='yellow', marker='o', s=20,
                           edgecolors='red', linewidth=1, alpha=0.8,
                           label=f'Hot Cluster ({len(hot_pixels)} pixels)')

                # Highlight the best hot pixel (first in list)
                if len(hot_pixels) > 0:
                    best_hot_x = hot_pixels[0][1]
                    best_hot_y = hot_pixels[0][0]
                    ax1.scatter(best_hot_x, best_hot_y, c='red', marker='o', s=150,
                               edgecolors='white', linewidth=3, alpha=1.0,
                               label=f'Hot Best (Ts={anchor_result.hot_cluster.ts_median:.1f}K)')

            # Add legend
            ax1.legend(loc='upper left', fontsize=8, framealpha=0.9,
                      bbox_to_anchor=(0.02, 0.98), ncol=1)
        
        # Fallback: Plot single anchor pixels from calibration_result
        elif calibration_result is not None:
            cold_x = getattr(calibration_result, 'cold_pixel_x', None)
            cold_y = getattr(calibration_result, 'cold_pixel_y', None)
            hot_x = getattr(calibration_result, 'hot_pixel_x', None)
            hot_y = getattr(calibration_result, 'hot_pixel_y', None)

            if cold_x is not None and cold_y is not None:
                ax1.scatter(cold_x, cold_y, c='blue', marker='o', s=50, edgecolors='white',
                           linewidth=1.5, alpha=0.9, label='Cold Pixel')

            if hot_x is not None and hot_y is not None:
                ax1.scatter(hot_x, hot_y, c='red', marker='o', s=50, edgecolors='white',
                           linewidth=1.5, alpha=0.9, label='Hot Pixel')

            ax1.legend(loc='upper left', fontsize=8, framealpha=0.8, bbox_to_anchor=(0.02, 0.98))

        # Panel 2: ETa Map
        ax2 = axes[0, 1]
        if 'ET_daily' in cube.data:
            et_data = cube.data['ET_daily'].values
            et_masked = np.ma.masked_invalid(et_data)
            
            im = ax2.imshow(et_masked, cmap=self._get_colormap('ETa'), origin='upper')
            plt.colorbar(im, ax=ax2, shrink=0.8, label='ETa (mm/day)')
            ax2.set_title('Daily ET Map')
        else:
            ax2.text(0.5, 0.5, 'ET data not available',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Daily ET Map (Not Available)')
        
        ax2.set_xlabel('Pixel X')
        ax2.set_ylabel('Pixel Y')
        
        # Panel 3: ETrF Map
        ax3 = axes[1, 0]
        if 'ETrF' in cube.data:
            etrf_data = cube.data['ETrF'].values
            etrf_masked = np.ma.masked_invalid(etrf_data)
            
            # Get dynamic vmax from data
            valid_etrf = etrf_data[~np.isnan(etrf_data)]
            etrf_vmax = np.nanmax(valid_etrf) if len(valid_etrf) > 0 else 2.0
            
            im = ax3.imshow(etrf_masked, cmap=self._get_colormap('ETrF'),
                          vmin=0, vmax=etrf_vmax, origin='upper')
            plt.colorbar(im, ax=ax3, shrink=0.8, label='ETrF')
            ax3.set_title('Reference ET Fraction')
        else:
            ax3.text(0.5, 0.5, 'ETrF data not available',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('ETrF Map (Not Available)')
        
        ax3.set_xlabel('Pixel X')
        ax3.set_ylabel('Pixel Y')
        
        # Panel 4: Energy Balance Pie Chart
        ax4 = axes[1, 1]
        try:
            if all(b in cube.data for b in ['R_n', 'G', 'H', 'LE']):
                Rn_mean = np.nanmean(cube.data['R_n'].values)
                G_mean = np.nanmean(cube.data['G'].values)
                H_mean = np.nanmean(cube.data['H'].values)
                LE_mean = np.nanmean(cube.data['LE'].values)
                
                total = G_mean + H_mean + LE_mean
                if total > 0:
                    proportions = [G_mean/total * 100, H_mean/total * 100, LE_mean/total * 100]
                    labels = [f'G ({proportions[0]:.1f}%)',
                             f'H ({proportions[1]:.1f}%)',
                             f'LE ({proportions[2]:.1f}%)']
                    colors = ['#A23B72', '#F18F01', '#C73E1D']
                    
                    wedges, texts, autotexts = ax4.pie(
                        [G_mean, H_mean, LE_mean],
                        labels=labels,
                        colors=colors,
                        autopct='',
                        startangle=90
                    )
                    ax4.set_title(f'Energy Balance\n(Rn = {Rn_mean:.1f} W/m²)')
            else:
                ax4.text(0.5, 0.5, 'Energy balance data\nnot available',
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Energy Balance (Not Available)')
        except Exception as e:
            ax4.text(0.5, 0.5, f'Error: {str(e)}',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Energy Balance (Error)')
        
        # Add scene info
        scene_id = cube.metadata.get('scene_id', 'Unknown')
        date_str = cube.acquisition_time.strftime('%Y-%m-%d') if cube.acquisition_time else 'Unknown'
        
        fig.suptitle(f'METRIC ET Summary - {scene_id} ({date_str})',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return self._save_figure(fig, output_path, show)
    
    def plot_quality_mask(
        self,
        quality_data: Union[np.ndarray, xr.DataArray],
        cube: DataCube,
        output_path: Optional[str] = None,
        show: bool = False,
        title: str = "Quality Mask"
    ) -> Optional[str]:
        """Create quality mask visualization.
        
        Args:
            quality_data: Quality mask array
            cube: DataCube containing metadata
            output_path: Output file path
            show: Whether to display the plot
            title: Plot title
            
        Returns:
            Output file path if saved
        """
        # Convert to numpy array
        if isinstance(quality_data, xr.DataArray):
            quality_data = quality_data.values
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create custom colormap for quality
        colors_qa = ['red', 'yellow', 'green', 'blue', 'gray']
        cmap_qa = mcolors.ListedColormap(colors_qa)
        
        # Mask invalid values
        quality_masked = np.ma.masked_invalid(quality_data)
        
        im = ax.imshow(
            quality_masked,
            cmap=cmap_qa,
            vmin=0,
            vmax=4,
            origin='upper'
        )
        
        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1, 2, 3, 4])
        cbar.set_label('Quality Class')
        cbar.ax.set_yticklabels([
            'Cloud', 'Cloud Shadow', 'Water', 'Vegetation', 'Other'
        ])
        
        ax.set_title(title)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        
        plt.tight_layout()
        
        return self._save_figure(fig, output_path, show)
