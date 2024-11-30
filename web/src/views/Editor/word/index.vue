<script setup>
import { onUnmounted, ref, onMounted } from "vue";
import { createClient } from "@liveblocks/client";
import { LiveblocksYjsProvider } from "@liveblocks/yjs";
import * as Y from "yjs";
import { useEditor, EditorContent } from "@tiptap/vue-3";
import StarterKit from "@tiptap/starter-kit";
import Collaboration from "@tiptap/extension-collaboration";
import CollaborationCursor from "@tiptap/extension-collaboration-cursor";

const leave = ref(null);
const editorContentRef = ref(null);

// Set up Liveblocks client
const client = createClient({
  publicApiKey: "pk_dev_RN3lh0rKwGDhDeUf13-gMQnUaUBzGp14NByC1rn9TEYYUCEMr0oPWLtKEp9dxk6x",
});

// Enter a multiplayer room
const info = client.enterRoom("my-room");
const room = info.room;
leave.value = info.leave;

// Set up Yjs document and Liveblocks Yjs provider
const yDoc = new Y.Doc();
const yProvider = new LiveblocksYjsProvider(room, yDoc);

let editor;

onMounted(() => {
  // Set up the Tiptap editor
  editor = useEditor({
    element: editorContentRef.value,
    extensions: [
      StarterKit.configure({
        // The Collaboration extension comes with its own history handling
        history: false,
      }),
      // Register the Yjs document with Tiptap
      Collaboration.configure({
        document: yDoc,
      }),
      CollaborationCursor.configure({
        provider: yProvider,
      }),
    ],
  });
});

onUnmounted(() => {
  leave?.();
  editor?.destroy();
});
</script>

<template>
  <div ref="editorContentRef">
    <editor-content :editor="editor" />
  </div>
</template>
