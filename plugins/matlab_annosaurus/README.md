# MATLAB_ANNOSAURUS

[Annosaurus](https://github.com/underwatervideo/annosaurus) is MBARI's new video annotation service. It manages the organization, storage and retrieval of image and video annotations. It's API is exposed via REST/JSON. MBARI is in the process of building tools around annosaurus.

This module is a demo on how to store output from your detectors in Annosaurus' data store. To run a demo:


1. Start Annosaurus
    - Start [Docker](https://www.docker.com/)
    - Start annosaurus with `docker run --name=annosaurus -p 8080:8080 hohonuuli/annosaurus`
        - This starts an in-memory database and REST server. The data will be lost when Annosaurus is stopped.
        - If you need permanent data-store, refer to [DEPLOYMENT.md](https://github.com/underwatervideo/annosaurus/blob/master/DEPLOYMENT.md).
2. Start Matlab
3. cd to this directory in Matlab
4. run `mock`. This will put some data into Annosaurus and return it as the `annotation_data` global variable in Matlab.


## Caveats

This is a work in progress. VIAME does not yet pass all parameters needed for effective data archiving. Currently missing is:
    - uuid: An [universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)  used to group images together.
    - image path: Need full image path passed in order to correctly resolve which image was being analyzed. VIAME currently passes a name only, without the path.
