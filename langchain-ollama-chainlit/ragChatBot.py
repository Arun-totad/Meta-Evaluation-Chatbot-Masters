from typing import List
import PyPDF2 
from io import BytesIO
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
#from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain_community.llms import Ollama

from langchain_community.chat_models import ChatOllama

from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl
import time

# RecursiveCharacterTextSplitter object with specific chunk size and overlap parameters for splitting text into smaller chunks that the LLM’s context window can handle and store it in a vector database.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


# The decorator function on_chat_start() will be called when the chat session is started from chainLit.
@cl.on_chat_start
async def on_chat_start(): #async : a function that can suspend execution to be resumed later
    pdf_file = None

    # Prompting the user to upload a PDF file and waits for the user's response. It ensures that the uploaded file is in PDF format and meets the size criteria.
    while pdf_file is None:
        pdf_file = await cl.AskFileMessage(
            content="Howdy! \n I'm a chatbot with LLM capabilities and can summarize any given pdf. Test me out! :) \n\n Kindly upload a pdf file to begin Meta-Evaluation!",
            accept=["application/pdf"], #Only accepts PDF files
            max_size_mb=20, #size criteria
            timeout=180, #Timeout in seconds if uploading fails
        ).send()
 #Add a error console for the user without pdf!

    #Start time for pdf processing
    pdfUploadStart = time.time()

    # Retrieving the first pdf file uploaded by the user, and sending a message indicating that the file is being processed.
    # print(pdf_file)
    file = pdf_file[0]

    msg = cl.Message(content=f"Processing `{file.name}` .......!")
    await msg.send()


    # Reading the PDF file and converting it into text using PyPDF2 library
    pdf_text = ""
    pdf = PyPDF2.PdfReader(file.path)
    for page in pdf.pages:
        pdf_text += page.extract_text()
    
    # print(pdf_text)
    # print("pdf_text")
    texts = text_splitter.split_text(pdf_text) # Spliting the text into chunks

    # print(texts)
    # print("texts")

    # Create a metadata for each chunk and chroma vector store
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    embeddings = OllamaEmbeddings(model="llama2") #Calling the LLM llama2 model running locally
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )


    # Storing the processed information as chat history in memory for futher processing
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )


    # Creatiing a chain that uses the Chroma vector store to retrieve and process the information in the pdf]
    # LCEL - check out it
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="llama2"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    #Stop time for pdf processing
    pdfUploadEnd = time.time()
    pdfProcessingTime = int(pdfUploadEnd - pdfUploadStart)
    if(pdfProcessingTime > 60):
        minutes = pdfProcessingTime // 60
        seconds = pdfProcessingTime - (minutes * 60)
        pdfProcessingTime = str(minutes) + " minutes "  + str(seconds) + " seconds!"
    else:
         pdfProcessingTime = str(pdfProcessingTime) + " seconds!"
    
    # Letting the user know that the system is ready to process LLM on the uploaded document.
    msg.content = f"Processing `{file.name}` is completed successfully. Performing Meta-Evaluation now. \n Time taken to process the pdf -> " + pdfProcessingTime
    await msg.update()

    cl.user_session.set("chain", chain)

    message = "THE ACCURACY STANDARDS ARE INTENDED TO ENSURE THAT AN EVALUATION EMPLOYS SOUND THEORY, DESIGNS, METHODS, AND REASONING IN ORDER TO MINIMIZE INCONSISTENCIES, DISTORTIONS, AND MISCONCEPTIONS AND PRODUCE AND REPORT TRUTHFUL EVALUATION FINDINGS AND CONCLUSIONS.THE ACCURACY STANDARDS ARE INTENDED TO ENSURE THAT AN EVALUATION EMPLOYS SOUND THEORY, DESIGNS, METHODS, AND REASONING IN ORDER TO MINIMIZE INCONSISTENCIES, DISTORTIONS, AND MISCONCEPTIONS AND PRODUCE AND REPORT TRUTHFUL EVALUATION FINDINGS AND CONCLUSIONS."
    msg = cl.Message(content=message)   
    await msg.send()

    messages = ["***A1 JUSTIFIED CONCLUSIONS AND DECISIONS*** \n Please respond each of the statements with 'true' or 'false'? \n A11. Can you address each contracted evaluation question based on information that is sufficiently broad, deep, reliable, contextually relevant, culturally sensitive, and valid? \n A12. Can you derive defensible conclusions that respond to the evaluation’s stated purposes, e.g., to identify and assess the program’s strengths and weaknesses, main effects and side effects, and worth and merit? \n A13.Can you limit conclusions to the applicable time periods, contexts, purposes, and activities? \n A14. Can you identify the persons who determined the evaluation’s conclusions, e.g., the evaluator using the obtained information plus inputs from a broad range of stakeholders? \n A15. Can you identify and report all important assumptions, the interpretive frameworks and values employed to derive the conclusions, and any appropriate caveats? \n A16. Can you report plausible alternative explanations of the findings and explain why rival explanations were rejected?", "***A2 Valid Information*** \n Please respond each of the statements with 'true' or 'false'? \n A21. Through communication with the full range of stakeholders develop a coherent, widely understood set of concepts and terms needed to assess and judge the program within its cultural context ? \n A22. Assure—through such means as systematic protocols, training, and calibration--that data collectors competently obtain the needed data ? \n A23. Document the methodological steps taken to protect validity during data selection, collection, storage, and analysis ? \n A24. Involve clients, sponsors, and other stakeholders sufficiently to ensure that the scope and depth of interpretations are aligned with their needs and widely understood ? \n A25. Investigate and report threats to validity, e.g., by examining and reporting on the merits of alternative explanations ? \n A26. Assess and report the comprehensiveness, quality, and clarity of the information provided by the procedures as a set in relation to the information needed to address the evaluation’s purposes and questions ?"] 
    # messages = ["***A1 JUSTIFIED CONCLUSIONS AND DECISIONS*** \n Please respond each of the statements with 'true' or 'false'? \n A11. Can you address each contracted evaluation question based on information that is sufficiently broad, deep, reliable, contextually relevant, culturally sensitive, and valid? \n A12. Can you derive defensible conclusions that respond to the evaluation’s stated purposes, e.g., to identify and assess the program’s strengths and weaknesses, main effects and side effects, and worth and merit? \n A13.Can you limit conclusions to the applicable time periods, contexts, purposes, and activities? \n A14. Can you identify the persons who determined the evaluation’s conclusions, e.g., the evaluator using the obtained information plus inputs from a broad range of stakeholders? \n A15. Can you identify and report all important assumptions, the interpretive frameworks and values employed to derive the conclusions, and any appropriate caveats? \n A16. Can you report plausible alternative explanations of the findings and explain why rival explanations were rejected?", "***A2 Valid Information*** \n Please respond each of the statements with 'true' or 'false'? \n A21. Through communication with the full range of stakeholders develop a coherent, widely understood set of concepts and terms needed to assess and judge the program within its cultural context ? \n A22. Assure—through such means as systematic protocols, training, and calibration--that data collectors competently obtain the needed data ? \n A23. Document the methodological steps taken to protect validity during data selection, collection, storage, and analysis ? \n A24. Involve clients, sponsors, and other stakeholders sufficiently to ensure that the scope and depth of interpretations are aligned with their needs and widely understood ? \n A25. Investigate and report threats to validity, e.g., by examining and reporting on the merits of alternative explanations ? \n A26. Assess and report the comprehensiveness, quality, and clarity of the information provided by the procedures as a set in relation to the information needed to address the evaluation’s purposes and questions ?"] 
    
    finalEvalDict = []
    i = 0
    for message in messages:
        i = i + 1
        replyUploadStart = time.time()

        msg = cl.Message(content=message)   
        await msg.send()

        # This line retrieves the conversational retrieval chain object from the user session using the key "chain". It assumes that the chain object has been previously set in the user session.
        chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
        
        # Instance of AsyncLangchainCallbackHandler, which is used to handle asynchronous callbacks during language chain operations.
        cb = cl.AsyncLangchainCallbackHandler()

        # Invoking the conversational retrieval chain (chain) with the content of the received message (message.content). It awaits the result, which includes the response and any associated source documents. It also specifies the callback handler (cb) to handle asynchronous operations.
        res = await chain.ainvoke(message, callbacks=[cb])

        # Extracting the response from LLM and the associated source documents (document/trained model) from the result obtained after invoking the chain.
        answer = res["answer"] #Stored as a string
        source_documents = res["source_documents"]  # source type: List[Document]

        # print(answer)

        import re
        number_regex = r"\d+"
        results = re.split(number_regex, answer)

        scoreFlag = 0

        for result in results:
            # print("------------------------")
            # print(result)

            condition1 = result.find("True")
            condition2 = result.find("Yes")
            # print("condition index for score:" + str(condition))
            # print(type(condition))

            if (condition1 > 0 or condition2 > 0):
                scoreFlag = scoreFlag + 1  
                print("LLM found a match as true!")

            # elif result.find("Partially true") > -1:
            #     scoreFlag = scoreFlag + 0.5 #confirm for value
                
            else:
                print("LLM match not found !")
                continue  

            print("Current Evaluation score: " + str(scoreFlag))
            print("-----------------------------------------")
        
        if (scoreFlag == 6):
            evalScore = "EXCELLENT"
        elif (scoreFlag == 5):
            evalScore = "VERY GOOD"
        elif (scoreFlag == 4):
            evalScore = "GOOD"
        elif (scoreFlag == 2 or scoreFlag == 3):
            evalScore = "FAIR"
        else:
            evalScore = "POOR"    
        
        finalEvalDict.append(evalScore)

        # Empty array to store source references
        text_elements = []  # type: List[cl.Text]

        # Checkinf if there are any source documents associated with the response. If there are, it iterates over each source document, creates a text element for it, and appends it to the text_elements list. It also updates the response to include information about the sources.
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            replyUploadEnd = time.time()
            replyProcessingTime = int(replyUploadEnd - replyUploadStart)
            if(replyProcessingTime > 60):
                minutes = replyProcessingTime // 60
                seconds = replyProcessingTime - (minutes * 60)
                replyProcessingTime = str(minutes) + " minutes "  + str(seconds) + " seconds!"
            else:
                replyProcessingTime = str(replyProcessingTime) + " seconds!"
                
            if source_names:
                answer += f"\n\nSources: {', '.join(source_names)}" + "\n Response time from ChatBot -> " + replyProcessingTime +"\n\n Evaluation score for A" + str(i) +" : "+str(scoreFlag)  + "\n Final evaluation for the above metaEvaluation ->"+ evalScore
            else:
                answer += "\nNo sources found" + "\n Response time from ChatBot -> " + replyProcessingTime
        
        await cl.Message(content=answer, elements=text_elements).send()

    print(finalEvalDict)
    
    display = "***Meta-Evaluation is completed successfully with total of 12 criteria for two sub criteria of Accuracy!*** \n"
    # Scoring the Evaluation for ACCURACY
    if (finalEvalDict.count("EXCELLENT") >= 0):
        reviewScore = finalEvalDict.count("EXCELLENT")
        excCntRate = (reviewScore * 4)
        print("Number of EXCELLENT ratings (0-4) -> " + str(reviewScore) +" x 4 = " + str(excCntRate))
        display = display + ("Number of EXCELLENT ratings (0-4) -> " + str(reviewScore) +" x 4 = " + str(excCntRate)) + "\n"
        
    if (finalEvalDict.count("VERY GOOD") >= 0):
        reviewScore = finalEvalDict.count("VERY GOOD")
        vGoodCntRate = (reviewScore * 3)
        print("Number of VERY GOOD ratings (0-4) -> " + str(reviewScore) +" x 3 = " + str(vGoodCntRate))
        display = display + ("Number of VERY GOOD ratings (0-4) -> " + str(reviewScore) +" x 3 = " + str(vGoodCntRate)) + "\n"
        
    if (finalEvalDict.count("GOOD") >= 0):
        reviewScore = finalEvalDict.count("GOOD")
        goodCntRate = (reviewScore * 2)
        print("Number of GOOD ratings (0-4) -> " + str(reviewScore) +" x 2 = " + str(goodCntRate))
        display = display + ("Number of GOOD ratings (0-4) -> " + str(reviewScore) +" x 2 = " + str(goodCntRate)) + "\n"

    if (finalEvalDict.count("FAIR") >= 0):
        reviewScore = finalEvalDict.count("FAIR")
        reviewCntRate = (reviewScore * 1)
        print("Number of FAIR ratings (0-4) -> " + str(reviewScore) +" x 1 = " + str(reviewCntRate))
        display = display + ("Number of FAIR ratings (0-4) -> " + str(reviewScore) +" x 1 = " + str(reviewCntRate)) + "\n"

        
    totalScoreRating = excCntRate + vGoodCntRate + goodCntRate + reviewCntRate
    print("Total Score rating for Accuracy ->" +str(totalScoreRating))
    display = display + ("***Total Score rating for Accuracy*** -> " +str(totalScoreRating))

    message = display
    msg = cl.Message(content=message)   
    await msg.send()





# The function main() and decorator that will be called when a message is prompted from the user during the chat session.
@cl.on_message
async def main(message: cl.Message):
    print("message")
    print(message)
    
    replyUploadStart = time.time()

    # This line retrieves the conversational retrieval chain object from the user session using the key "chain". It assumes that the chain object has been previously set in the user session.
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    
    # Instance of AsyncLangchainCallbackHandler, which is used to handle asynchronous callbacks during language chain operations.
    cb = cl.AsyncLangchainCallbackHandler()

    # Invoking the conversational retrieval chain (chain) with the content of the received message (message.content). It awaits the result, which includes the response and any associated source documents. It also specifies the callback handler (cb) to handle asynchronous operations.
    res = await chain.ainvoke(message.content, callbacks=[cb])

    # Extracting the response from LLM and the associated source documents (document/trained model) from the result obtained after invoking the chain.
    answer = res["answer"]
    source_documents = res["source_documents"]  # source type: List[Document]

    # Empty array to store source references
    text_elements = []  # type: List[cl.Text]

    # Checkinf if there are any source documents associated with the response. If there are, it iterates over each source document, creates a text element for it, and appends it to the text_elements list. It also updates the response to include information about the sources.
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        replyUploadEnd = time.time()
        replyProcessingTime = int(replyUploadEnd - replyUploadStart)
        if(replyProcessingTime > 60):
            minutes = replyProcessingTime // 60
            seconds = replyProcessingTime - (minutes * 60)
            replyProcessingTime = str(minutes) + " minutes "  + str(seconds) + " seconds!"
        else:
            replyProcessingTime = str(replyProcessingTime) + " seconds!"
            
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}" + "\n Response time from ChatBot -> " + replyProcessingTime
        else:
            answer += "\nNo sources found" + "\n Response time from ChatBot -> " + replyProcessingTime

    # Sending a response to the UI along with any associated text elements (source documents) to the user. The content parameter contains the response text, and the elements parameter contains any additional text elements.
    await cl.Message(content=answer, elements=text_elements).send()
