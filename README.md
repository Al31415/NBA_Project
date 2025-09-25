# NBA Game Winner Prediction Using Graph Neural Networks

This repository contains code and data for predicting the winner of NBA games using **Graph Neural Networks (GNNs)**. Players and their interactions during a game are modeled as a **heterogeneous graph** to capture intricate relationships and predict game outcomes effectively.

---

## Overview

### What is this project about?

The goal is to accurately predict NBA game winners using player statistics and interactions represented through graph
structures. A heterogeneous graph model is employed, using:

- **Nodes**: Players from both the home and visitor teams and the teams themselves.
- **Edges**: Player interactions (assists, blocks, steals, fouls, and rebounds) as well as edges between the players and their team. 

---

## Accuracy
Achieves **80% accuracy** when predicting game outcomes from just the **first half** of play-by-play data with optimizations to come to get that higher!

---

## Getting Started

###  Installation

Clone this repository and install the necessary packages:

```bash
git clone https://github.com/Al31415/NBA_Project.git
cd NBA_Project
pipenv install
pipenv shell

```
---

### Data

Ensure the following CSV files are in the data directory:

- `nba_regular_season_playbyplay_data.csv`
- `team_ids.csv`

> These contain the raw play-by-play data and team identifiers. They can be collected using the data handling scripts.

---

### Training and Evaluation

To train the Graph Neural Network model, simply run:

```bash
python main.py
```
## License

This project is licensed under the **MIT License**

---

##  Contact

For questions, suggestions, or feedback, feel free to reach out:

**Alagu Thiagarajan**  
[alaguthiagarajan123@gmail.com](mailto:alaguthiagarajan123@gmail.com)

---
