from logger import logger 

class llm_explainer:
    def __init__(self,llm_client):
        #llm_client can be OpenAI, Azure OpenAI, or any LLM wrapper
        self.llm=llm_client
        logger.info("genai explainer initialised")
     
     
    def explain(self,original_text:str,sentiment:str,score:str)->str:
        """Generate a human readable explaination for model prediction"""
        
        prompt = f"""You are an assistant explaining the output of a machine learning model that analyzes medical drug reviews.
                    The ML model predicted the sentiment as: {sentiment},Confidence score (distance from decision boundary): {score:.3f}
                    Medical review:
                    \"\"\"{original_text}\"\"\"
                    Explain in simple terms WHY the model likely made this prediction.
                    Focus on:
                    - Benefit-related words
                    - Side-effect mentions
                    - Overall tone
                    Do NOT reclassify the sentiment.
                    Do NOT mention machine learning jargon."""
        logger.info("seeding prompt to llm for explaination")
        response=self.llm.generate(prompt)
        logger.info("recieved explaination from llm")
        
        return response.strip()
    
class DummyLLMClient:
    def generate(self, prompt: str) -> str:
        return (
            "The review mentions noticeable improvement in symptoms, "
            "such as reduced pain or relief, which signals effectiveness. "
            "Although minor side effects may be present, the overall tone "
            "leans positive, leading the model to classify it as positive."
        )

        