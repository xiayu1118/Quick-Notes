<template>
  <DocumentEditor
    id="docEditor"
    documentServerUrl="http://documentserver/"
    :config="config"
    :events_onDocumentReady="onDocumentReady"
    :onLoadComponentError="onLoadComponentError"
  />
</template>

<script lang="ts">
import { defineComponent } from "vue";
import { DocumentEditor } from "@onlyoffice/document-editor-vue";

export default defineComponent({
  name: "ExampleComponent",
  components: {
    DocumentEditor,
  },
  data() {
    return {
      config: {
        document: {
          fileType: "docx",
          key: "Khirz6zTPdfd7",
          title: "Example Document Title.docx",
          url: "https://example.com/url-to-example-document.docx",
        },
        documentType: "word",
        editorConfig: {
          callbackUrl: "https://example.com/url-to-callback.ashx",
        },
      },
    };
  },
  methods: {
    onDocumentReady() {
      console.log("Document is loaded");
    },
    onLoadComponentError(errorCode: any, errorDescription: any) {
      switch (errorCode) {
        case -1: // Unknown error loading component
          console.log(errorDescription);
          break;

        case -2: // Error load DocsAPI from http://documentserver/
          console.log(errorDescription);
          break;

        case -3: // DocsAPI is not defined
          console.log(errorDescription);
          break;
      }
    },
  },
});
</script>
