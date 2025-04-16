from langchain_community.tools.tavily_search import TavilySearchResults

def get_profile_url_tavily(text: str, mock_tavily : bool = True) -> str:
    """Extracts the profile URL from the text using Tavily Search API.
    
    Args:
        text (str): The input text containing the profile URL.
        
    Returns:
        str: The extracted profile URL or an empty string if not found.
    """
    if mock_tavily:
        tavily_res = """[{'title': 'Eden Marco | Open Data Science Conference', 'url': 'https://odsc.com/blog/speaker/eden-marco/', 'content': 'Eden is a seasoned backend software engineer with deep expertise in generative AI, cloud, and cybersecurity. With years of experience in backend development', 'score': 0.7752821}, {'title': 'Eden Marco - Google | LinkedIn', 'url': 'https://www.linkedin.com/in/eden-marco', 'content': 'Eden Marco\nLLMs @ Google Cloud | Best-selling Udemy Instructor | Backend & GenAI | Opinions stated here are my own, not those of my company\nUnited States, United States of America\n4995 connections, 6872 followers\n\n\nAbout:\nBackend developer --> Now an AI Engineer, Udemy.com best seller instructor\nOpinions stated here are my own, not those of my company\n\n\nExperience:\nLLM Specialist, Customer Engineering - Google Cloud at Google (https://www.linkedin.com/company/google)\nJul 2023 - Present\n\n\nEducation:\nTechnion - Israel Institute of Technology\nBachelor’s Degree, Computer Science\nJan 2015 - Dec 2019\nGrade: N/A\nActivities and societies: N/A', 'score': 0.60386235}, {'title': 'Highlights by Eden Marco (@EdenEmarco177) / X', 'url': 'https://twitter.com/EdenEmarco177/highlights', 'content': "Software developer/Programmer/Software engineer. Joined July 2015. 433 Following · 3,090 Followers · Posts · Replies · Highlights · Media. Eden Marco's", 'score': 0.5786631}, {'title': 'Eden Marco - Strategy Analyst - Deloitte Digital - LinkedIn', 'url': 'https://www.linkedin.com/in/edenmarco', 'content': 'Eden Marco\nStrategy Analyst at Deloitte\nNew York, United States\n1032 connections, 1036 followers\n\n\nAbout:\nN/A\n\n\nExperience:\nStrategy Analyst at Deloitte Digital (https://www.linkedin.com/company/deloitte-digital)\nAug 2023 - Present\nNew York, New York, United States\n\n\nEducation:\nUNC Kenan-Flagler Business School\nBSBA, Business Administration\nN/A - Present\nGrade: N/A\nActivities and societies: N/A', 'score': 0.55767727}, {'title': 'Eden MARCO | U of T | Department of Medicine | Research profile', 'url': 'https://www.researchgate.net/profile/Eden-Marco', 'content': 'Eden MARCO | Cited by 11 | of University of Toronto, Toronto (U of T) | Read 6 publications | Contact Eden MARCO.', 'score': 0.55362654}]"""
    else:
        # Initialize TavilySearchResults with the text
        tavily_search = TavilySearchResults()
        
        # Extract the profile URL
        tavily_res = tavily_search.run(f"{text}")

        # print("Tavily Search Results:\n")
        # print(tavily_res)
        # print("\nTavily Search Over\n")
    return tavily_res