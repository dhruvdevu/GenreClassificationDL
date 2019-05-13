# Understanding Music by extracting song embeddings and classifying genre using Deep Learning
## Introduction
Consider the following 3 songs:
<iframe src="https://open.spotify.com/embed/track/6or1bKJiZ06IlK0vFvY75k" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
<br>
<iframe src="https://open.spotify.com/track/6fxVffaTuwjgEk5h9QyRjy" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
<br>
<iframe src="https://open.spotify.com/track/4fzsfWzRhPawzqhX8Qt9F3" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
<br>
How do we determine which songs are similar and which are not alike? As humans, we use a variety of distinguishing factors such as artist, genre, tone, etc, based on information from lyrics and popularity, as well as musical information like rhythm, scale, timbre, pitch, chord progressions, and more. Computer systems that can process and understand music in this manner hold great value to music producers and consumers alike. Companies like Google, Apple, Spotify, Pandora, and dozens of others are all interested in retrieving information from music that would allow them to make better recommendations, and understand what types of music and listeners belong together. Historically, this information has been obtained from user and usage data-centric approaches [NEEDS CITATION]. We aim, however, to extract this information using audio features and lyrics of songs using deep learning. Specifically, we develop a model to classify songs by genre, and to generate a latent embedding representation for each song, which we use to cluster songs and which can be used as a gauge for song similarity using cosine distance.


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ddhruv97/GenreClassificationBlog/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
