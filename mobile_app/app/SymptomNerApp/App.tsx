/**
 * App.tsx
 *
 * Single-screen entry point for the Symptom NER mobile app. Runs the V05
 * BioBERT model on-device to extract affirmed (POS) / denied (NEG) symptom
 * mentions from clinical free-text.
 *
 * Single screen, top to bottom:
 *   - Disclaimer header
 *   - Clinical-note text input + Run button (on-device inference)
 *   - Detected-symptoms output (POS/NEG spans)
 *   - Collapsible, searchable list of the 893 trained symptoms
 */

import { useMemo, useRef, useState } from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  useColorScheme,
  View,
} from 'react-native';
import {
  SafeAreaProvider,
  SafeAreaView,
} from 'react-native-safe-area-context';
import symptoms from './assets/symptoms.json';
import { predictSpans } from './src/ner/infer';
import type { Span } from './src/ner/aggregate';
import { encode } from './src/ner/tokenizer';
import { MODEL_CONFIG, VOCAB } from './src/ner/modelConfig';

// Model accepts MODEL_CONFIG.maxSeqLen tokens TOTAL including [CLS] and [SEP].
// Cap user content at maxSeqLen - 2 so we never exceed it.
const MAX_CONTENT_TOKENS = MODEL_CONFIG.maxSeqLen - 2;

/** Counts the model-side content tokens for `text` ([CLS]/[SEP] excluded). */
function countContentTokens(text: string): number {
  return Math.max(0, encode(text, VOCAB).ids.length - 2);
}

/** A single trained symptom: ontology id + its preferred label. */
type Symptom = {
  id: string;
  prefLabel: string;
};

const SYMPTOMS: Symptom[] = symptoms as Symptom[];

/** Root component: wires up safe-area + status bar around the single screen. */
function App(): React.JSX.Element {
  const isDarkMode = useColorScheme() === 'dark';

  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <SafeAreaView style={styles.safeArea}>
        <MainScreen />
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

/** The single screen: disclaimer, input, output placeholder, symptom list. */
function MainScreen(): React.JSX.Element {
  const [note, setNote] = useState<string>('');
  const [symptomsExpanded, setSymptomsExpanded] = useState<boolean>(false);
  const [query, setQuery] = useState<string>('');
  const [spans, setSpans] = useState<Span[] | null>(null);
  const [running, setRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  // Monotonic counter to ignore stale inference results when the user re-runs.
  const runIdRef = useRef<number>(0);

  /** Live token count of the current note (content only; excludes specials). */
  const tokenCount = useMemo<number>(() => countContentTokens(note), [note]);

  /**
   * Accepts a new note only if it fits within the model's token budget. Once
   * the limit is reached, further keystrokes / pastes that would exceed it are
   * rejected (the previous text is preserved).
   */
  const onChangeNote = (next: string): void => {
    if (countContentTokens(next) <= MAX_CONTENT_TOKENS) {
      setNote(next);
    }
  };

  /** Runs on-device NER over the current note and stores the detected spans. */
  const onRun = async (): Promise<void> => {
    const myRunId = ++runIdRef.current;
    setRunning(true);
    setError(null);
    try {
      const result = await predictSpans(note);
      if (myRunId !== runIdRef.current) {
        return; // a newer run has started — discard this stale result
      }
      // CONFLICT (and any unexpected) spans are not shown in the UI per product
      // decision; log them so the developer can still inspect them in Metro.
      const conflicts = result.filter(
        s =>
          s.label !== 'O' &&
          s.label !== 'SYMPTOM_POS' &&
          s.label !== 'SYMPTOM_NEG',
      );
      if (conflicts.length > 0) {
        console.warn('[NER] Spans hidden from UI:', conflicts);
      }
      setSpans(result);
    } catch (err: unknown) {
      if (myRunId !== runIdRef.current) {
        return;
      }
      setError(String(err));
      setSpans(null);
    } finally {
      if (myRunId === runIdRef.current) {
        setRunning(false);
      }
    }
  };

  // Only affirmed/denied entities are shown in the UI; 'O' is structural and
  // CONFLICT is logged to the console (see onRun).
  const entities = useMemo<Span[]>(
    () =>
      (spans ?? []).filter(
        s => s.label === 'SYMPTOM_POS' || s.label === 'SYMPTOM_NEG',
      ),
    [spans],
  );

  /** Symptoms whose prefLabel contains the query (case-insensitive). */
  const filteredSymptoms = useMemo<Symptom[]>(() => {
    const q = query.trim().toLowerCase();
    if (q === '') {
      return SYMPTOMS;
    }
    return SYMPTOMS.filter(s => s.prefLabel.toLowerCase().includes(q));
  }, [query]);

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView
        style={styles.flex}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        {/* Region 1: welcome + disclaimer */}
        <View style={styles.section}>
          <Text style={styles.title}>Symptom NER</Text>
          <Text style={styles.disclaimer}>
            For learning purposes only. Not for medical use.
          </Text>
        </View>

        {/* Region 2: clinical-note input */}
        <View style={styles.section}>
          <Text style={styles.label}>Clinical note</Text>
          <TextInput
            style={styles.input}
            value={note}
            onChangeText={onChangeNote}
            placeholder="Type the clinical note here..."
            placeholderTextColor="#9ca3af"
            multiline
            textAlignVertical="top"
          />
          <Text
            style={
              tokenCount >= MAX_CONTENT_TOKENS
                ? styles.tokenCountLimit
                : styles.tokenCount
            }
          >
            {tokenCount} / {MAX_CONTENT_TOKENS} tokens
            {tokenCount >= MAX_CONTENT_TOKENS ? ' (limit reached)' : ''}
          </Text>
          <Pressable
            style={[
              styles.runButton,
              (running || note.trim() === '') && styles.runButtonDisabled,
            ]}
            disabled={running || note.trim() === ''}
            onPress={onRun}
          >
            <Text style={styles.runButtonText}>
              {running ? 'Analyzing...' : 'Run'}
            </Text>
          </Pressable>
        </View>

        {/* Region 3: detected symptom spans */}
        <View style={styles.section}>
          <Text style={styles.label}>Detected symptoms</Text>
          <View style={styles.outputBox}>
            {error !== null ? (
              <Text style={styles.errorText}>{error}</Text>
            ) : spans === null ? (
              <Text style={styles.placeholderText}>
                Enter a note and tap Run.
              </Text>
            ) : entities.length === 0 ? (
              <Text style={styles.placeholderText}>No symptoms detected.</Text>
            ) : (
              entities.map((s, i) => (
                <View key={`${s.start}-${i}`} style={styles.entityRow}>
                  <View
                    style={[
                      styles.badge,
                      s.label === 'SYMPTOM_POS' ? styles.badgePos : styles.badgeNeg,
                    ]}
                  >
                    <Text style={styles.badgeText}>
                      {s.label === 'SYMPTOM_POS' ? 'POS' : 'NEG'}
                    </Text>
                  </View>
                  <Text style={styles.entityText}>{s.text}</Text>
                </View>
              ))
            )}
          </View>
        </View>

        {/* Region 4: collapsible trained-symptom list */}
        <View style={styles.section}>
          <Pressable
            style={styles.collapsibleHeader}
            onPress={() => setSymptomsExpanded(prev => !prev)}
          >
            <Text style={styles.collapsibleTitle}>
              List of symptoms the model was trained with (total of{' '}
              {SYMPTOMS.length})
            </Text>
            <Text style={styles.chevron}>{symptomsExpanded ? '▲' : '▼'}</Text>
          </Pressable>
          {symptomsExpanded && (
            <View style={styles.collapsibleBody}>
              <TextInput
                style={styles.searchInput}
                value={query}
                onChangeText={setQuery}
                placeholder="Search symptoms..."
                placeholderTextColor="#9ca3af"
                autoCapitalize="none"
                autoCorrect={false}
              />
              <Text style={styles.resultCount}>
                {filteredSymptoms.length} of {SYMPTOMS.length}
              </Text>
              <FlatList
                style={styles.symptomList}
                data={filteredSymptoms}
                keyExtractor={item => item.id}
                nestedScrollEnabled
                keyboardShouldPersistTaps="handled"
                renderItem={({ item }) => (
                  <Text style={styles.symptomRow}>{item.prefLabel}</Text>
                )}
                ListEmptyComponent={
                  <Text style={styles.placeholderText}>No matches.</Text>
                }
              />
            </View>
          )}
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  content: {
    padding: 16,
    gap: 20,
  },
  section: {
    gap: 8,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#111827',
  },
  disclaimer: {
    fontSize: 13,
    fontStyle: 'italic',
    color: '#b45309',
  },
  label: {
    fontSize: 15,
    fontWeight: '600',
    color: '#374151',
  },
  input: {
    minHeight: 120,
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
    color: '#111827',
  },
  outputBox: {
    minHeight: 80,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 8,
    padding: 12,
    backgroundColor: '#f9fafb',
  },
  placeholderText: {
    fontSize: 14,
    color: '#6b7280',
  },
  errorText: {
    fontSize: 13,
    color: '#b91c1c',
  },
  tokenCount: {
    fontSize: 12,
    color: '#6b7280',
    textAlign: 'right',
  },
  tokenCountLimit: {
    fontSize: 12,
    color: '#b91c1c',
    textAlign: 'right',
  },
  runButton: {
    backgroundColor: '#2563eb',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
  },
  runButtonDisabled: {
    backgroundColor: '#9ca3af',
  },
  runButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  entityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
    gap: 8,
  },
  badge: {
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
    minWidth: 38,
    alignItems: 'center',
  },
  badgePos: {
    backgroundColor: '#16a34a',
  },
  badgeNeg: {
    backgroundColor: '#dc2626',
  },
  badgeText: {
    color: '#ffffff',
    fontSize: 11,
    fontWeight: '700',
  },
  entityText: {
    flex: 1,
    fontSize: 15,
    color: '#111827',
  },
  collapsibleHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    padding: 12,
  },
  collapsibleTitle: {
    flex: 1,
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
  },
  chevron: {
    fontSize: 12,
    color: '#6b7280',
    marginLeft: 8,
  },
  collapsibleBody: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 8,
  },
  searchInput: {
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 14,
    color: '#111827',
  },
  resultCount: {
    fontSize: 12,
    color: '#6b7280',
  },
  symptomList: {
    maxHeight: 280,
  },
  symptomRow: {
    fontSize: 14,
    color: '#374151',
    paddingVertical: 6,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#e5e7eb',
  },
});

export default App;
