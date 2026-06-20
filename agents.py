import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

class CactusNewsRoom:
    def __init__(self):
        # 1. Load hidden API keys from the .env file
        load_dotenv()

        # 2. Initialize the LLM and the search tool
        self.llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.7,
        )


        self.search_tool = SerperDevTool()

        # 3. Define the Agents as instance attributes
        self.wire_scout = Agent(
            role="Global Wire Journalist",
            goal="Find weird, deeply ironic, or highly specific true news stories from around the world.",
            backstory=(
                "A cynical, coffee-fueled investigative journalist who spends 24 hours a day "
                "refreshing local news sites, looking for the absurd realities of human existence."
            ),
            tools=[self.search_tool],
            verbose=True,
            llm=self.llm
        )

        self.satirist = Agent(
            role="Chief Satirical Editor",
            goal="Invert real-world headlines into brilliant, biting, Onion-style satirical premises.",
            backstory=(
                "A veteran comedy writer who views the world through a lens of profound irony. "
                "Expert at applying 'The Onion' formula: taking an ordinary, small piece of news and "
                "treating it like a global, apocalyptic crisis (Rule of Contrast). Conversely, if a "
                "massive, serious event occurs, you treat it like a minor, boring bureaucratic inconvenience. "
                "You always punch up at systems of power, corporate greed, or general human silliness."
            ),
            verbose=True,
            llm=self.llm,
        )

        self.staff_writer = Agent(
            role="Satirical Staff Writer",
            goal="Write gripping, journalistic parody articles based on approved pitches.",
            backstory=(
                "A deadpan writer who crafts articles that sound exactly like a real Associated Press report. "
                "You possess zero self-awareness—you never break character, never use emojis, and never 'wink' "
                "at the camera to indicate a joke. The humor comes entirely from how seriously you treat the "
                "fabricated reality. You are a master of the standard journalistic formula."
            ),
            verbose=True,
            llm=self.llm
        )

        # 4. Define the Tasks
        self.fetch_news_task = Task(
            description=(
                "Search the web for 3 strange, ironic, or deeply bureaucratic news events that happened recently. "
                "Look for odd local council disputes, bizarre tech updates, or peculiar human interest stories."
            ),
            expected_output="A list of 3 real-world news summaries with sources.",
            agent=self.wire_scout
        )

        self.pitch_satire_task = Task(
            description=(
                "Review the real news stories provided by the scout. Select the single best story and pivot "
                "it into an Onion-style satirical headline. Apply the Rule of Contrast strictly. "
                "If a tech company changed an app icon, make it sound like a terrifying cultural paradigm shift. "
                "If a massive global shift happened, treat it like a mild annoyance."
            ),
            expected_output="A single brilliant satirical headline and a 2-sentence explanation of the comedic angle.",
            agent=self.satirist
        )

        self.write_article_task = Task(
            description=(
                "Take the approved satirical headline and write a full 300-word news article.\n\n"
                "CRITICAL RULES FOR WRITING:\n"
                "1. TONALITY: Maintain an unwavering, dry, professional journalistic prose style throughout.\n"
                "2. ZERO SELF-AWARENESS: Absolutely no emojis, no puns, and no meta-commentary. Keep a straight face.\n"
                "3. THE FAKE EVERYMAN QUOTE: You must include at least two quotes from fake 'local residents', "
                "'bystanders', or 'generic experts'. Format them exactly like a real paper: "
                "'“It’s just devastating,” said local resident John B., staring blankly at a wall.'\n"
                "4. STRUCTURE: Start with a clear headline, a dateline (e.g., OMAHA, NE—), an inverted pyramid structure, "
                "and conclude with an absurdly mundane closing thought."
            ),
            expected_output="A beautifully formatted Markdown news article with a headline, dateline, and body paragraphs.",
            agent=self.staff_writer
        )

        # 5. Assemble the Crew
        self.cactus_network = Crew(
            agents=[self.wire_scout, self.satirist, self.staff_writer],
            tasks=[self.fetch_news_task, self.pitch_satire_task, self.write_article_task],
            process=Process.sequential,
        )

    def generate_article(self) -> str:
        """Kicks off the crew workflow and returns the generated article text."""
        # result.raw returns the raw string content of the final task output
        result = self.cactus_network.kickoff()
        return result.raw


# ==========================================
# HOW TO USE THIS CLASS IN OTHER SCRIPTS:
# ==========================================
if __name__ == "__main__":
    # Instantiate the newsroom object
    newsroom = CactusNewsRoom()
    
    # Generate the article and capture it in a variable
    article_content: str = newsroom.generate_article()
    
    # You now have the string ready to be sent to a database, frontend web app, or email!
    print("Article captured successfully! Character count:", len(article_content))

    ## save to a text file 
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(article_content)