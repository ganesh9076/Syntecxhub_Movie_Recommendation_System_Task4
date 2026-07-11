# Movie Recommender — Evaluation Report

## 1. Qualitative Evaluation — Sample Queries

For each seed movie, we show the top-5 results from the **content-based** model (similar genres/tags) and the **item-based collaborative filtering** model (similar rating patterns).

### Seed movie: *Toy Story*

**Content-based (metadata similarity):**

| clean_title                            | genres                                      |   similarity |
|:---------------------------------------|:--------------------------------------------|-------------:|
| A Bug's Life                           | Adventure|Animation|Children|Comedy         |       0.8434 |
| Toy Story 2                            | Adventure|Animation|Children|Comedy|Fantasy |       0.8136 |
| Antz                                   | Adventure|Animation|Children|Comedy|Fantasy |       0.7536 |
| The Adventures of Rocky and Bullwinkle | Adventure|Animation|Children|Comedy|Fantasy |       0.7536 |
| The Emperor's New Groove               | Adventure|Animation|Children|Comedy|Fantasy |       0.7536 |

**Item-based collaborative filtering (rating-pattern similarity):**

| clean_title                        |   similarity |
|:-----------------------------------|-------------:|
| Toy Story 2                        |       0.5726 |
| Jurassic Park                      |       0.5656 |
| Independence Day (a.k.a. ID4)      |       0.5643 |
| Star Wars: Episode IV - A New Hope |       0.5574 |
| Forrest Gump                       |       0.5471 |

### Seed movie: *The Matrix*

**Content-based (metadata similarity):**

| clean_title                        | genres                 |   similarity |
|:-----------------------------------|:-----------------------|-------------:|
| Lawnmower Man 2: Beyond Cyberspace | Action|Sci-Fi|Thriller |       0.6046 |
| Screamers                          | Action|Sci-Fi|Thriller |       0.6046 |
| Johnny Mnemonic                    | Action|Sci-Fi|Thriller |       0.6046 |
| Timecop                            | Action|Sci-Fi|Thriller |       0.6046 |
| Solo                               | Action|Sci-Fi|Thriller |       0.6046 |

**Item-based collaborative filtering (rating-pattern similarity):**

| clean_title                                    |   similarity |
|:-----------------------------------------------|-------------:|
| Fight Club                                     |       0.7139 |
| Star Wars: Episode V - The Empire Strikes Back |       0.7009 |
| Saving Private Ryan                            |       0.6796 |
| Star Wars: Episode IV - A New Hope             |       0.6634 |
| Star Wars: Episode VI - Return of the Jedi     |       0.661  |

### Seed movie: *The Dark Knight*

**Content-based (metadata similarity):**

| clean_title                             | genres                           |   similarity |
|:----------------------------------------|:---------------------------------|-------------:|
| Need for Speed                          | Action|Crime|Drama|IMAX          |       0.8077 |
| Batman Begins                           | Action|Crime|IMAX                |       0.7787 |
| Fast Five (Fast and the Furious 5, The) | Action|Crime|Drama|Thriller|IMAX |       0.7529 |
| Eagle Eye                               | Action|Crime|Thriller|IMAX       |       0.7223 |
| A Good Day to Die Hard                  | Action|Crime|Thriller|IMAX       |       0.7223 |

**Item-based collaborative filtering (rating-pattern similarity):**

| clean_title                                   |   similarity |
|:----------------------------------------------|-------------:|
| Inception                                     |       0.7273 |
| Iron Man                                      |       0.6705 |
| The Dark Knight Rises                         |       0.6661 |
| Batman Begins                                 |       0.6513 |
| The Lord of the Rings: The Return of the King |       0.6206 |

### Seed movie: *Forrest Gump*

**Content-based (metadata similarity):**

| clean_title                                                     | genres                   |   similarity |
|:----------------------------------------------------------------|:-------------------------|-------------:|
| Train of Life (Train de vie)                                    | Comedy|Drama|Romance|War |       0.5542 |
| Tiger and the Snow, The (La tigre e la neve)                    | Comedy|Drama|Romance|War |       0.5542 |
| I Served the King of England (Obsluhoval jsem anglického krále) | Comedy|Drama|Romance|War |       0.5542 |
| Life Is Beautiful (La Vita è bella)                             | Comedy|Drama|Romance|War |       0.5213 |
| Colonel Chabert, Le                                             | Drama|Romance|War        |       0.5211 |

**Item-based collaborative filtering (rating-pattern similarity):**

| clean_title              |   similarity |
|:-------------------------|-------------:|
| The Shawshank Redemption |       0.713  |
| Jurassic Park            |       0.6883 |
| Pulp Fiction             |       0.6855 |
| Braveheart               |       0.6431 |
| The Silence of the Lambs |       0.6395 |

### Seed movie: *The Shawshank Redemption*

**Content-based (metadata similarity):**

| clean_title                                   | genres      |   similarity |
|:----------------------------------------------|:------------|-------------:|
| The Green Mile                                | Crime|Drama |       0.6414 |
| Shanghai Triad (Yao a yao yao dao waipo qiao) | Crime|Drama |       0.5247 |
| Hate (Haine, La)                              | Crime|Drama |       0.5247 |
| The Young Poisoner's Handbook                 | Crime|Drama |       0.5247 |
| New Jersey Drive                              | Crime|Drama |       0.5247 |

**Item-based collaborative filtering (rating-pattern similarity):**

| clean_title              |   similarity |
|:-------------------------|-------------:|
| Forrest Gump             |       0.713  |
| Pulp Fiction             |       0.7024 |
| The Silence of the Lambs |       0.6471 |
| The Usual Suspects       |       0.6318 |
| Schindler's List         |       0.6291 |


## 2. Personalized Recommendations (Matrix Factorization CF)

For a few sample users, we show 3 movies they rated highly (their taste) next to the model's top-5 predicted recommendations from movies they haven't seen.

### User 1

- Rated highly: Seven (a.k.a. Se7en), The Usual Suspects, Bottle Rocket

- Recommended:

|   movieId | clean_title                |   predicted_rating |
|----------:|:---------------------------|-------------------:|
|       541 | Blade Runner               |               4.62 |
|      1653 | Gattaca                    |               4.59 |
|       778 | Trainspotting              |               4.58 |
|       858 | The Godfather              |               4.57 |
|       589 | Terminator 2: Judgment Day |               4.55 |

### User 45

- Rated highly: Ocean's Thirteen, The Usual Suspects, Pirates of the Caribbean: The Curse of the Black Pearl

- Recommended:

|   movieId | clean_title              |   predicted_rating |
|----------:|:-------------------------|-------------------:|
|       318 | The Shawshank Redemption |               5    |
|       527 | Schindler's List         |               4.64 |
|      1089 | Reservoir Dogs           |               4.38 |
|        47 | Seven (a.k.a. Se7en)     |               4.37 |
|     44191 | V for Vendetta           |               4.33 |

### User 234

- Rated highly: Toy Story, Powder, Pocahontas

- Recommended:

|   movieId | clean_title                         |   predicted_rating |
|----------:|:------------------------------------|-------------------:|
|       457 | The Fugitive                        |               4.07 |
|       589 | Terminator 2: Judgment Day          |               4.07 |
|      2716 | Ghostbusters (a.k.a. Ghost Busters) |               3.95 |
|      2762 | The Sixth Sense                     |               3.88 |
|       150 | Apollo 13                           |               3.88 |


## 3. Quantitative Evaluation — Matrix Factorization RMSE

Held-out RMSE (lower is better) of predicted vs actual star rating, on an 80/20 per-user train/test split (80896 train / 19940 test ratings):

|   n_factors |     RMSE |
|------------:|---------:|
|          10 | 0.922507 |
|          30 | 0.928215 |
|          60 | 0.936827 |


_For reference, always predicting the global mean rating (~3.5) typically gives an RMSE around 1.0-1.05 on this dataset, so this is our naive baseline._

- Baseline (predict global mean 3.50): RMSE = 1.0493
