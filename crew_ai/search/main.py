from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()
tool = SerperDevTool(
    locale="en",
    location="Bengaluru, India",
    n_results=2,
)

print(tool.run(search_query="who is cv raman"))

# Using Tool: Search the internet

# Search results: Title: Jeux Olympiques de Paris 2024 - Actualités, calendriers, résultats
# Link: https://olympics.com/fr/paris-2024
# Snippet: Quels sont les sports présents aux Jeux Olympiques de Paris 2024 ? · Athlétisme · Aviron · Badminton · Basketball · Basketball 3x3 · Boxe · Breaking · Canoë ...
# ---
# Title: Billetterie Officielle de Paris 2024 - Jeux Olympiques et Paralympiques
# Link: https://tickets.paris2024.org/
# Snippet: Achetez vos billets exclusivement sur le site officiel de la billetterie de Paris 2024 pour participer au plus grand événement sportif au monde.
# ---
