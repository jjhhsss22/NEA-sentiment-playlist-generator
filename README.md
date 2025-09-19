# sentiment-playlist-generator

This project explores the intersection of machine learning and music recommendation.
It uses sentiment analysis to classify text input from the user into different emotions, and then generates a playlist that the gradually transitions from the detected emotion to the user's emotion of choice.

A dataset of 1,000 songs is mapped onto a 2D Valence–Arousal graph, allowing a graph traversal algorithm to find a smooth emotional path between the starting and target moods.

🧠 **Modeling**: trained on a labeled dataset of emotions using neural networks.

🎵 **Application**: maps predicted sentiment → emotion-sensitive playlist generation.

💡 **Goal**: demonstrate how AI can bridge natural language understanding and personalized music experiences, including potential applications in music therapy.

Here is the full documentation of the project: https://docs.google.com/document/d/1xRGxF-19MLz0YQcSxIe7rRYlSt83DojQSZsl5oeYK54/edit?usp=sharing


(the dataset used couldn't be uploaded due to its size)


✨ **Potential Improvements**  
(Now that I have more experience)
- Expand dataset to more songs.

- Add support for real-time streaming services (Spotify/YouTube).

- Improve emotion prediction with transformers or contextual/positional embeddings.

- Add user feedback loop to refine playlist suggestions.
