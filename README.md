🏀 NBA Game Winner Prediction Using Graph Neural Networks

This repository contains code and data for predicting the winner of NBA games using Graph Neural Networks (GNNs). Players and their interactions during a game are modeled as a heterogeneous graph to capture intricate relationships and predict game outcomes effectively.

⸻

📌 Overview

What is this project about?

The goal is to accurately predict NBA game winners using player statistics and interactions represented through graph structures. A heterogeneous graph model is employed, using:
	•	Nodes: Players from both the home and visitor teams
	•	Edges: Player interactions (assists, blocks, steals, fouls, and rebounds)

⸻

🌟 Features
	•	✅ Graph Representation: Clearly captures complex player interactions using heterogeneous graphs
	•	🚀 Advanced ML Techniques: Utilizes Graph Neural Networks (GNNs) for enhanced predictive performance
	•	🔍 Interpretable Results: Enables deeper insights into key interactions that influence game outcomes
	•	📈 High Accuracy: Achieves 80% accuracy when predicting game outcomes from just the first half of play-by-play data

⸻

⚙️ Getting Started

🧱 Installation

Clone this repository and install the necessary packages:

git clone https://github.com/Al31415/NBA_Project.git
cd NBA_Project
pipenv install
pipenv shell



⸻

📂 Data

Ensure the following CSV files are in the root directory:
	•	nba_regular_season_playbyplay_data.csv
	•	team_ids.csv

These contain the raw play-by-play data and team identifiers. They can be collected using the data handling scripts.

⸻

🏃‍♂️ Training and Evaluation

To train the Graph Neural Network model, simply run:

python main.py

This script will:
	•	Load and preprocess the data
	•	Build a heterogeneous graph
	•	Train the GNN model
	•	Report validation accuracy

⸻

🎯 Accuracy

The current model achieves 80% accuracy in predicting NBA game outcomes using only first-half play-by-play data.

⸻

🧹 Key Components
	•	gcn_nba_winner_hetero.py: Defines the heterogeneous GNN model
	•	Nodes: Players from home and visitor teams
	•	Edges: Assists, blocks, steals, fouls, and rebounds
	•	Output: Binary prediction — home win or not

⸻

📊 Results

The model demonstrates strong performance, showing the effectiveness of modeling player interactions as a graph for game outcome prediction.

⸻

🤝 Contributing

Feel free to fork the repository, experiment, and submit pull requests with:
	•	🚧 Bug fixes
	•	🌱 Feature improvements
	•	🧪 New analyses

⸻

📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

⸻

📬 Contact

For questions, suggestions, or feedback, please contact:
Alagu Thiagarajan
📧 alaguthiagarajan123@gmail.com

⸻
