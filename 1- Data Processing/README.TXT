Data Processing Phase
The data processing phase is a critical component of the project, serving as the foundation for organizing and structuring information effectively for the Retrieval-Augmented Generation (RAG) system. 
This phase ensures that the data is systematically segmented to enhance retrieval efficiency and maintain consistency throughout the workflow.

1- Initial Data Segmentation:
The data processing phase begins with a hierarchical segmentation approach. The first level of segmentation is based on the year, ensuring that documents from different periods are properly categorized.
2- Bulletin Number Categorization:
Within each year, the data is further divided based on the bulletin number. This step helps maintain logical grouping and allows for efficient retrieval of related documents.
3- File Index Breakdown:
Each bulletin is then segmented by the index within its respective file. This granular breakdown ensures that individual sections of the documents are easily accessible and structured appropriately.
4- Storing Structured Data:
After the segmentation process is completed, the structured data is stored in a JSON file located in the Data folder. This JSON file preserves the hierarchical organization and serves as an intermediary format for subsequent processing steps.
5- Transition to the Next Stage:
The processed data in JSON format is then passed to the next stage of the workflow, enabling efficient retrieval and utilization within the Retrieval-Augmented Generation (RAG) system.

This structured approach ensures that the information is logically organized, easy to access, and optimally prepared for use in later stages of the project.
