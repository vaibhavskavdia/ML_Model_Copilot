from ML_Model_Copilot.logger import logger 

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
        if "Negative" in prompt:
            return (
                "The review emphasizes worsening symptoms and adverse side effects, "
                "such as increased pain or discomfort, which strongly influenced the "
                "model to classify the sentiment as negative."
            )
        else:
            return (
                "The review highlights symptom relief and therapeutic benefits, "
                "indicating that the medication was effective overall, leading the "
                "model to classify the sentiment as positive."
            )


class GENAIExplainer:
    def __init__(self,llm_client):
        self.llm_client=llm_client
        
    def explain(self,text:str,sentiment:str,score:float)->str:
        prompt=self._build_prompt(text,sentiment,score)
        
        return self.llm_client.generate(prompt)
    
    def _build_prompt(self,text:str,sentiment:str,score:float)->str:
        return (
            f"Text: {text}\n"
            f"Predicted Sentiment: {sentiment}\n"
            f"Model Score: {score:.3f}\n\n"
            f"Explain in simple terms why this sentiment was predicted."
        )
