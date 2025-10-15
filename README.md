# Operator-approach-to-epidemic-models

This repository contains the Python code and resources for the Master's Thesis, "Operator approach to epidemic systems in networks," presented at the Universidad Nacional de Colombia in 2022. The research explores epidemic dynamics using an operator-based mathematical framework.

The core results, including simulations and data visualizations, are available as Jupyter Notebooks. The full thesis is accessible through the Universidad Nacional de Colombia's official repository: [https://repositorio.unal.edu.co/](https://repositorio.unal.edu.co/).

## Repository Structure

The repository is organized as follows:

-   `/notebooks`: Contains the main Jupyter Notebooks with various simulations and analyses, such as:
    -   SIRS models with exponential holding times (`EXP HT SIRS.ipynb`)
    -   Reaction-diffusion process generators (`Reaction_generator.ipynb`)
    -   Analysis of the Whittle no-outbreak probability (`Whittle no outbreak probability.ipynb`)
-   `/Threshold_SIR`: Includes Python scripts and resulting data (`.csv` files) related to the analysis of the SIR model's epidemic threshold.
-   `/Graficas_modelos_sencillos`: Notebooks for plotting and visualizing simpler epidemic models.
-   `/Presentaciones`: Slide decks created during the development of this work, summarizing the key results.

## Getting Started

### Prerequisites

To run the code in this repository, you will need a Python environment. We recommend using the [Anaconda](https://www.anaconda.com/products/distribution) distribution.

The following Python packages are required:
-   `numpy`
-   `scipy`
-   `seaborn`
-   `matplotlib`
-   `jupyter`

### Installation & Usage

1.  Clone the repository to your local machine:
    ```
    git clone https://github.com/joarojasve/Operator-approach-to-epidemic-models.git
    ```
2.  Navigate to the cloned directory and install the required packages. If you are using `pip`, you can run:
    ```
    pip install numpy scipy seaborn matplotlib jupyter
    ```
3.  Launch Jupyter Lab or Jupyter Notebook:
    ```
    jupyter lab
    ```
4.  Navigate to the desired notebook file and execute the cells to reproduce the results.

## Contact

For any questions or inquiries regarding this work, please feel free to reach out to `joarojasve@unal.edu.co`.
