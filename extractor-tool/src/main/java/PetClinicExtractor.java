import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class PetClinicExtractor {

    private static final String SOURCE_ROOT = "../spring-petclinic/src/main/java";//"C:\\MSc Software Architecture\\Project\\spring-petclinic\\src\\main\\java";
    private static final String OUTPUT_FILE = "petclinic_data.json";

    public static void main(String[] args) throws IOException {
        System.out.println("Starting extraction from: " + SOURCE_ROOT);

        // 1. Configure Symbol Solver (Critical for Call Graph extraction)
        // This allows the parser to "understand" types across different files
        CombinedTypeSolver combinedTypeSolver = new CombinedTypeSolver();
        combinedTypeSolver.add(new ReflectionTypeSolver()); // Solves JDK classes (String, List)
        combinedTypeSolver.add(new JavaParserTypeSolver(new File(SOURCE_ROOT))); // Solves PetClinic classes

        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(combinedTypeSolver);
        StaticJavaParser.getConfiguration().setSymbolResolver(symbolSolver);

        List<FileEntry> extractedData = new ArrayList<>();

        // 2. Walk through all .java files in the project
        Files.walk(Paths.get(SOURCE_ROOT))
                .filter(path -> path.toString().endsWith(".java"))
                .forEach(path -> {
                    try {
                        extractFromFile(path, extractedData);
                    } catch (Exception e) {
                        System.err.println("Skipping file " + path.getFileName() + ": " + e.getMessage());
                    }
                });

        // 3. Save the results to JSON
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(new File(OUTPUT_FILE), extractedData);

        System.out.println("Extraction complete. Data saved to " + OUTPUT_FILE);
    }

    private static void extractFromFile(Path path, List<FileEntry> results) throws IOException {
        CompilationUnit cu = StaticJavaParser.parse(path);

        cu.findAll(ClassOrInterfaceDeclaration.class).forEach(c -> {
            FileEntry entry = new FileEntry();
            entry.fileName = path.getFileName().toString();
            entry.className = c.getNameAsString();
            entry.packageName = cu.getPackageDeclaration().map(p -> p.getNameAsString()).orElse("");

            // --- PART A: Token Extraction (For HDP-LDA) ---
            // This builds the "bag of words" for topic modelling

            // 1. Class Name tokens
            entry.tokens.addAll(tokenize(entry.className));

            // 2. Method Names & Parameters
            c.getMethods().forEach(m -> {
                entry.tokens.addAll(tokenize(m.getNameAsString()));
                m.getParameters().forEach(p -> entry.tokens.addAll(tokenize(p.getNameAsString())));
            });

            // 3. Field (Variable) Names
            c.getFields().forEach(f -> {
                f.getVariables().forEach(v -> entry.tokens.addAll(tokenize(v.getNameAsString())));
            });

            // --- PART B: Structural Extraction (For Call Graphs) ---
            // This finds dependencies for structural validation
            c.accept(new VoidVisitorAdapter<Void>() {
                @Override
                public void visit(MethodCallExpr n, Void arg) {
                    super.visit(n, arg);
                    try {
                        ResolvedMethodDeclaration resolved = n.resolve();
                        String targetClass = resolved.getClassName();

                        // Filter: We only want internal calls (PetClinic logic), not Java/Spring calls
                        if (!targetClass.startsWith("java.") && !targetClass.startsWith("org.springframework")) {
                            entry.methodCalls.add(targetClass + "." + resolved.getName());
                        }
                    } catch (Exception e) {
                        // Symbol solver might fail on some libs; safe to ignore for research prototype
                    }
                }
            }, null);

            results.add(entry);
        });
    }

    // Helper: Splits "OwnerController" -> ["owner", "controller"] and lowercases them
    private static List<String> tokenize(String identifier) {
        String[] parts = identifier.split("(?=[A-Z])");
        return Arrays.stream(parts)
                .map(String::toLowerCase)
                .filter(s -> s.length() > 1) // Remove single chars
                .collect(Collectors.toList());
    }

    // Data Structure for JSON output
    static class FileEntry {
        public String fileName;
        public String className;
        public String packageName;
        public List<String> tokens = new ArrayList<>();
        public Set<String> methodCalls = new HashSet<>();
    }
}